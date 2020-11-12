import subprocess

# mnist experiments
# python src/serverattack_main.py --model=dcgan --dataset=mnist --gpu=0  --epochs=36 --num_users=200 --frac=0.2 --local_ep=1 --wanted_label_index=8 --local_gan_epoch=50
# label gan_lr gan_epoch optim_method
local_gan_epoch = 25
local_epoch = 1
for label in range(10):
    for gan_lr in [0.0002, 0.002]:
                subprocess.call('python src/serverattack_main.py --mode=production --model=dcgan --dataset=mnist --gpu=0 --epochs=100 --num_users=200 --frac=0.2 --local_ep={} --wanted_label_index={} --local_gan_epoch={} --local_gan_lr={}'.format(local_epoch, label, local_gan_epoch, gan_lr), shell=True)
