# Nonlinear Optimizer for SLAM
SLAM에서 주로 다루어지는 전형화 된 문제들을 푸는 nonlinear optimizer 구현 모음

## 1. Installation
### Dependencies
#### Flann
```
sudo apt install libflann-dev
```
#### Ceres solver
```
sudo apt install libceres-dev
```
### Clone repository
```
git clone "https://github.com/ChanghyeonKim93/nonlinear_optimizer_for_slam.git"
cd nonlinear_optimizer_for_slam
mkdir build && cd build
cmake .. && make -j{$nproc}
```

## 2. Run Examples
Build 내부에 생성되는 executable들을 실행

## 3. Contents
#### Mahalanobis distance minimizer
* 3D NDT (Normal Distribution Transform)로 표현된 3D map 에 3D Point cloud를 정합하는 sensor pose를 찾는 문제

#### Reprojection error minimizer
* Reference camera frame 표현된 local 3D points와, 이에 상응하는 query camera frame에서 표현된 2D pixel points간의 reprojection error를 최소화하는 pose from reference frame to query frame을 찾는 문제

#### Pose graph optimizer
* Pose graph 내의 모든 pose가 상대 자세 제약조건을 최대한 만족하도록 최적화하는 문제 (including 2D , 3D cases)
