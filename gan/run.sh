# set to your dataroot location
DATA="data/"

git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git

# run dcgan
python dcgan.py $DATA

# run stylegan using original author's implementation
cd stylegan2-ada-pytorch
# cifar10, class conditioned to cats (label 3)
# seeds=0-35 will generate 36 different cat images
python generate.py --outdir=out --seeds=0-35 --class=3 --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl
# ffhq
python generate.py --outdir=out --trunc=1 --seeds=85 --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl