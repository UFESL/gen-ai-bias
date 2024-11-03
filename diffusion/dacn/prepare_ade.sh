# Use to prepare ADE Scene Parse dataset into format readable by HF datasets
# Only moves images around, thus only needs to be run once
DATA_DIR="data/ADEChallengeData2016"

cd $DATA_DIR
mkdir train
mkdir val
mkdir train/annotations
mkdir train/images
mkdir train/depth
mkdir val/annotations
mkdir val/images
mkdir val/depth

mv images/training/* train/images
mv images/validation/* val/images
mv annotations/training/* train/annotations
mv annotations/validation/* val/annotations

rm -rf images
rm -rf annotations
