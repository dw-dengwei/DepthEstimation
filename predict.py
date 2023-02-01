"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
from distutils.spawn import find_executable
from importlib.resources import read_binary
import os
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import util
from util.visualizer import save_images
from util import html
import cv2 as cv
import numpy as np
from PIL import Image
from util.util import tensor2im
from sklearn.metrics import mean_squared_error as mse_err
from data.base_dataset import get_transform, get_params

try:
    import wandb
except ImportError:
    print(
        'Warning: wandb package cannot be found. The option "--use_wandb" will result in error.'
    )


if __name__ == "__main__":
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = (
        -1
    )  # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(
        opt
    )  # create a dataset given opt.dataset_mode and other options
    model = create_model(
        opt
    )  # create a model given opt.model and other options
    model.setup(
        opt
    )  # regular setup: load and print networks; create schedulers

    # initialize logger
    model.eval()
    with torch.no_grad():
        for i in range(1, 100):
            if not os.path.exists(f'test_input/{i}.jpg'):
                continue
            img = Image.open(f'test_input/{i}.jpg').convert("RGB")
            mask = np.array(Image.open(f'mask/{i}.jpg'))[:, :, 0]
            mask[mask > 0] = 1
            mask = cv.resize(mask, (128, 128))
            mask = np.expand_dims(mask, axis=2)
            img = np.array(img)
            #img = img * mask
            img = Image.fromarray(img)
            transform_params = get_params(opt, img.size)
            transform = get_transform(opt, transform_params, grayscale=False)
            img = transform(img).unsqueeze(0)

            model.set_input({"A": img, "B": img, "A_paths": ""})
            model.forward()
            pred = model.fake_B.detach().cpu().numpy().squeeze()
            #pred = np.transpose(pred,(1,2,0))
            pred = cv.normalize(pred, None, 0, 255, cv.NORM_MINMAX)
            cv.imwrite(f'test_output/{i}.jpg', pred)
            print(f'test_output/{i}.jpg')
