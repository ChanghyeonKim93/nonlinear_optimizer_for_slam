# Nonlinear Optimizer for SLAM
SLAM에서 주로 다루어지는 전형화 된 문제들을 푸는 nonlinear optimizer 구현 모음

## 1. Installation
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
