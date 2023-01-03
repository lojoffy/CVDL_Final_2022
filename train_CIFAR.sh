# Recommend to train on 2 GPUs. Training NesT-T can use 1 GPU.
CUDA_VISIBLE_DEVICES=0,1 python main.py --config configs/cifar_nest.py2 --workdir="./checkpoints/nest_cifar_large"
