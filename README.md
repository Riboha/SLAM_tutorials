# Python robotics tutorial

<p align="center">
  <img width="40%" src="https://github.com/SeongboHa/SLAM_tutorials/blob/main/imgs/EKF_SLAM_gif.gif"/>
  <img width="40%" src="https://github.com/SeongboHa/SLAM_tutorials/blob/main/imgs/graph_SLAM_gif.gif"/>
</p>

## Contents
- [Localization](#localization)
- [SLAM](#slam)

## Install

```bash
conda create -n pyslam python==3.9
conda activate pyslam
pip install matplotlib numpy imageio pip install scipy scikit-learn PyQt5==5.14.2 pyqtgraph setuptools==58.2.0
sudo apt-get install libsuitesparse-dev libeigen3-dev

# build g2opy and install
conda activate pyslam
cd g2opy
mkdir build
cd build
cmake ..
make -j12
cd ..
python setup.py install
```

## Localization
<p align="center">
  <img width="40%" src="https://github.com/SeongboHa/SLAM_tutorials/blob/main/imgs/DR.png"/>
  <img width="40%" src="https://github.com/SeongboHa/SLAM_tutorials/blob/main/imgs/GPS.png"/>
</p>

<p align="center">
  <img width="40%" src="https://github.com/SeongboHa/SLAM_tutorials/blob/main/imgs/KF.png"/>
  <img width="40%" src="https://github.com/SeongboHa/SLAM_tutorials/blob/main/imgs/EKF.png"/>
  <img width="40%" src="https://github.com/SeongboHa/SLAM_tutorials/blob/main/imgs/PF.png"/>
</p>
ref : https://github.com/AtsushiSakai/PythonRobotics

### Kalman Filter
### Extended Kalman Filter
### Particle Filter

## SLAM
### EKF SLAM
ref : https://github.com/adarshmodh/EKF_SLAM/tree/master
```bash
conda activate pyslam
cd EKF_SLAM
python slam.py
```
### Pose-Graph SLAM
ref : https://github.com/goktug97/PyGraphSLAM
```bash
conda activate pyslam
cd PyGraphSLAM
python victoria_park.py
```