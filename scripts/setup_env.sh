
# Install MMPose and MMdet

## working versions:
### pytorch: 1.11.0
### cuda: 11.3
### mmcv-full: 1.5.0
### mmdet: 2.23.0
### numpy: 1.22.3
### cudnn: 8.2

# pytorch and cuda stuff
conda install -y pytorch=1.11.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch
sudo apt install -y nvidia-cuda-toolkit
conda install -y -c conda-forge cudnn

# mmpose
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
pip install openmim
mim install mmdet

# this is needed because of some weird numpy linking issues
pip install -U numpy

# tf and image processing
pip install piexif
pip install tqdm
pip install tensorflow
pip install focal-loss
conda install -y dask

# raise the number of files that can be opened
ulimit -n 4096

# to find libcudart.so
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/conda/envs/ai-badminton/lib/
