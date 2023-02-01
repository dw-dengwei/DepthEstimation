set -ex
python train.py \
--dataroot /home/dw/data/bu3dfe/augment \
--name bu3dfe \
--model pix2pix \
--netG unet_256 \
--direction AtoB \
--lambda_L1 100 \
--dataset_mode aligned \
--norm batch \
--pool_size 0 \
--use_wandb \
--gpu_ids 1,2 \
--batch_size 256 \
--print_freq 20 \
--display_freq 20 \
--lr 3e-4 \
--preprocess resize \
--load_size 256 \
--n_epochs 20 \
--n_epochs_decay 5 \
--lr_policy plateau \
--num_threads 8 \
--output_nc 1
