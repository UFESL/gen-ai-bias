DATA="data/"

# please download CelebA manually due to API limitations with Google Drive
# Align&Cropped Images: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
# download horse2zebra
wget -N http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/horse2zebra.zip -O $DATA/horse2zebra.zip
cd $DATA
unzip horse2zebra.zip
rm horse2zebra.zip