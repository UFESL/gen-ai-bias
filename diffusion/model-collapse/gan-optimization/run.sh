#!/bin/bash

clip="stochastic"
batch=20
niter=5001
size=500

# Loop through class indices
for ((class_idx=0; class_idx<10; class_idx++))
do
    python dcgan_reverse_mmd_real_latent.py --clip=$clip --batch=$batch --class_idx=$class_idx --niter=$niter --size=$size
done
