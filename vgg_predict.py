# Author: @Wei-Deng dw-dengwei@outlook.com

import torch.multiprocessing as mp
import numpy as np
import os
import torch
import cv2
import time
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from glob import glob
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from PIL import Image
from data.base_dataset import get_transform, get_params


img_size = 128
input_root = '/home/mnt/vgg/aligned'
seg_root = '/home/mnt/vgg/seg_show'
output_root = '/home/mnt/vgg/pix_depth'


class Data(Dataset):
    def __init__(self, file_list, seg_list, opt, device) -> None:
        self.file_list = file_list
        self.seg_list = seg_list
        self.device = device
        self.opt = opt

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        seg_path = self.seg_list[index]
        sample_id = get_sample_id(img_path)

        img = Image.open(img_path)
        mask = np.array(Image.open(seg_path))[:, :, 0]
        mask[mask > 0] = 1
        mask = cv2.resize(mask, (128, 128))
        mask = np.expand_dims(mask, axis=2)
        img = np.array(img)
        img = img * mask
        img = Image.fromarray(img)

        transform_params = get_params(self.opt, img.size)
        transform = get_transform(self.opt, transform_params, grayscale=False)
        img = transform(img).unsqueeze(0)

        return img, mask, sample_id


# @run_time
def save(depth, sample_id, id_output_root):
    save_path = os.path.join(
                    id_output_root, 
                    sample_id + '_depth.png'
                )
    if not os.path.exists(id_output_root):
        os.makedirs(id_output_root)

    return cv2.imwrite(save_path, depth)

        
def get_sample_id(fp):
    fp = fp.split('/')[-1]
    fp = fp.split('.')[0]
    return fp


def run(pid, *args):
    args = args[pid]
    ids = args[0]
    gpu_id = pid % 4
    device = torch.device(f'cuda:{gpu_id}')
    print(f'ids:{ids}\ngpu:{gpu_id}')
    model, opt = setup(gpu_id)
    for idx in ids:
        id_input_root = os.path.join(
            input_root,
            str(idx)
        )
        id_output_root = os.path.join(
            output_root,
            str(idx)
        )

        patterns = ('*.jpg', )
        img_list = []
        for p in patterns:
            img_list.extend(
                glob(os.path.join(id_input_root, p))
            )
        seg_list = list(map(
            lambda p: p.replace(input_root, seg_root),
            img_list
        ))
        dataset = Data(img_list, seg_list, opt, device)
        dataloader = DataLoader(
            dataset, 
            num_workers=15,
            pin_memory=False, 
        )
        for img, seg, sample_id in tqdm(dataloader, desc=str(pid) + '/' + str(idx)):
            sample_id = sample_id[0]
            img = img[0]
            seg = seg[0]
            depth = predict(img, seg, model, opt)
            save(depth, sample_id, id_output_root)


if __name__ == '__main__':
    pool = []
    n_proc = 4
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


def setup(gpu_id):
    opt = TestOptions().parse()  # get test options
    opt.gpu_ids = [gpu_id]
    model = create_model(
        opt
    )  # create a model given opt.model and other options
    model.setup(
        opt
    )  # regular setup: load and print networks; create schedulers
    #model.device = device
    #model.netG.to(device)

    # initialize logger
    model.eval()

    return model, opt


@torch.no_grad()
def predict(img, seg, model, opt):

    model.set_input({"A": img, "B": img, "A_paths": ""})
    # print(model.device, model.real_A.device)
    model.forward()
    pred = model.fake_B.detach().cpu().numpy().squeeze()
    pred = cv2.normalize(pred, None, 0, 255, cv2.NORM_MINMAX)
    pred = cv2.resize(pred, (128, 128))

    return pred
