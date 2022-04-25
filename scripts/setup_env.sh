
# Install MMPose and MMdet

## working versions:
### pytorch: 1.11.0
### cuda: 11.3
### mmcv-full: 1.5.0
### mmdet: 2.23.0
### numpy: 1.22.3
### cudnn: 8.2

conda install pytorch=1.11.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch

pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

pip install openmim
mim install mmdet

# this is needed because of some weird numpy linking issues
pip install -U numpy

conda install -c conda-forge cudnn

# raise the number of files that can be opened
ulimit -n 2048
