# GAN attack in Federated Learning

## Requirement:
- python 3.8.10
- pytorch 1.8.1

## How to run the attack:
1. Create a python 3.8.10 enviroment, we recommend create enviroment by [conda](https://docs.conda.io/en/latest/miniconda.html#)
```
conda create -n GANAttack python=3.8.10
conda activate GANAttack
```
2. Install requirement modules
```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirments.txt
```
3. Run
```
python src/serverattack_main.py --mode=production --model=dcgan --dataset=mnist --epochs=20 --num_users=100 --frac=0.1 --local_ep=1 --wanted_label_index=1 --local_gan_epoch=25 --local_gan_lr=0.002 --experiment_name='demo'
```
if you have a gpu, you can add argument `--gpu=0` to run in CUDA.