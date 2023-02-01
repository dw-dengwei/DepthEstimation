set -ex
python vgg_predict.py \
--dataroot /home/dw/data/bu3dfe/augment \
--name bu3dfe \
--model pix2pix \
--netG unet_256 \
--direction AtoB \
--dataset_mode aligned \
--norm batch \
--phase 'test' \
--num_test 1000 \
--output_nc 1 \
--gpu_ids 2
