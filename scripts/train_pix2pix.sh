set -ex
python train.py \
--dataroot /home/pris/dw/data/texas \
--name texas \
--model pix2pix \
--netG unet_256 \
--direction AtoB \
--lambda_L1 100 \
--dataset_mode aligned \
--norm batch \
--pool_size 0 \
--use_wandb \
--gpu_ids 0,1,2,3 \
--batch_size 16 \
--print_freq 20 \
--display_freq 20 \
--lr 3e-4 \
--n_epochs 300