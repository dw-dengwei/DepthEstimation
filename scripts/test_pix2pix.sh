set -ex
python test.py \
--dataroot /home/pris/dw/data/texas \
--name texas \
--model pix2pix \
--netG unet_256 \
--direction AtoB \
--dataset_mode aligned \
--norm batch \
--num_test 200
