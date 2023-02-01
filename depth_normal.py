# Author: @Wei-Deng dw-dengwei@outlook.com

import torch.multiprocessing as mp
import numpy as np
import os
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from glob import glob


img_size = 128
input_root = '/home/mnt/vgg/pix_depth'
output_root = '/home/mnt/vgg/pix_depth_normal'


class Depth(Dataset):
    def __init__(self, file_list) -> None:
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        fp = self.file_list[index]
        sample_id = get_sample_id(fp)
        depth = cv2.imread(fp, -1)
        return sample_id, depth


def get_normal(depth_map):
    """calculate normal map from depth map

    Args:
        depth_map (ndarray): depth map

    Returns:
        ndarray: normal map
    """
    d_im = depth_map.astype(np.float32)
    zy, zx = np.gradient(d_im)  
   
    normal = np.dstack((-zx, -zy, np.ones_like(d_im)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    # offset and rescale values to be in 0-255
    normal += 1
    normal /= 2
    # if show, comment
    normal *= 255
    # cv2.imwrite("normal.png", normal[:, :, ::-1])
    return normal[:,:,::-1]


# @run_time
def save(depth, sample_id, id_output_root):
    save_path = os.path.join(
                    id_output_root, 
                    sample_id + '_dzyx.png'
                )
    if not os.path.exists(id_output_root):
        os.makedirs(id_output_root)

    return cv2.imwrite(save_path, depth)

        
def get_sample_id(fp):
    fp = fp.split('/')[-1]
    fp = fp.split('.')[0]
    fp = fp.replace('_depth', '')
    return fp


def run(pid, *args):
    args = args[pid]
    ids = args[0]
    gpu_id = pid % 4
    device = torch.device(f'cuda:{gpu_id}')
    print(f'ids:{ids}\ngpu:{gpu_id}')
    for idx in ids:
        id_input_root = os.path.join(
            input_root,
            str(idx)
        )
        id_output_root = os.path.join(
            output_root,
            str(idx)
        )

        patterns = ('*.png', )
        file_list = []
        for p in patterns:
            file_list.extend(
                glob(os.path.join(id_input_root, p))
            )
        dataset = Depth(file_list)
        dataloader = DataLoader(
            dataset, 
            num_workers=15,
            pin_memory=False, 
        )
        for sample_id, depth in tqdm(dataloader, desc=str(pid) + '/' + str(idx)):
            sample_id = sample_id[0]
            depth = depth[0].numpy()
            normal = get_normal(depth)
            dzyx = np.concatenate([depth.reshape(128, 128, 1), normal], axis=2)
            save(dzyx, sample_id, id_output_root)


if __name__ == '__main__':
    pool = []
    n_proc = 10
    ids = [i for i in range(4000, 1 + 5000)]
    length = len(ids)
    step = int(length / n_proc) + 1

    args = []
    for i in range(0, length, step):
        args.append(
            (ids[i: i + step],)
        )
    args = tuple(args)
    mp.spawn(run, args, n_proc, join=True)
