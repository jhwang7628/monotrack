# TrackNetV2: Efficient TrackNet (GitLab)

The following is a modification of [TrackNetV2](https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2). First, download the data used to train tracknet net and place it in a directory of your choice. We'll refer to this directory as `data_dir`.

## :gear: 1. Install
### System Environment

- Ubuntu 18.04
- NVIDIA Gerfore GTX1080Ti
- Python3.5.2 / git / PYQT5 / OpenCV / pandas / numpy / PyMySQL
- TensorFlow 1.13.1/keras 2.2.4/Opencv 4.1.0/CUDA 10.1 (for TrackNetV2)

### Package
- First, you have to install cuda, cudnn and tensorflow, tutorial:
https://medium.com/@zhanwenchen/install-cuda-and-cudnn-for-tensorflow-gpu-on-ubuntu-79306e4ac04e
        
        $ sudo apt-get install git
        $ sudo apt-get install python3-pip
        $ pip3 install pyqt5
        $ pip3 install pandas
        $ pip3 install PyMySQL
        $ pip3 install opencv-python
        $ pip3 install imutils
        $ pip3 install Pillow
        $ pip3 install piexif
        $ pip3 install -U scikit-learn
        $ pip3 install keras
        $ git clone https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2
        
## :clapper: 2. Preparing the training data
Navigate to the `modified-tracknet` folder. Modify the `data_dir` variable in `generate_data.py` to the location of your downloaded data.
From within the folder, run the following:
`python3 generate-data.py --type=TRAIN`
`python3 generate-data.py --type=DISTILLATION`
`python3 generate-data.py --type=TEST`

## :hourglass_flowing_sand: 3. Distilation and normal training

### Distillation training
`python3 train_distill.py`

### Normal training (using distilled model as starting point)
`python3 train_faster.py --load_weights=tracknet_improved --save_weights=tracknet_improved_final`

## 4. Predictions
You can predict coordinate of shuttlecock for a single video with:

`python3 predict-mask.py --video_name=<videoPath> --load_weights=<weightPath>`



















