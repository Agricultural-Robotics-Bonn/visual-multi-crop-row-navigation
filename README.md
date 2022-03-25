       
# Visual Multi Crop Row Navigation in Arable Farming Fields      

<div align="center">
	<img src="data/motivation.png" alt="BonnBot" height="340" title="BonnBot"/>
</div>

Check out the [video1](https://youtu.be/z2Cb2FFZ2aU?t=43), of our robot following this approach to navigate on a real multi lace row-crop field (beans field).

### pyCUDA installation (optional)

	wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64$
	sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
	wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_install$
	sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
	sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
	sudo apt-get update
	sudo apt-get -y install cuda

### Dependencies
- OpenCV == 3.3.0.10
    - pip install opencv-python==3.3.0.10
    - pip install opencv-contrib-python==3.3.0.10
- ROS Melodic
- itertools
- scipy >= 1.5.2
- numpy >= 1.19.9

### Build and run

navigate to your catkin workspace folder, e.g.:
  
    cd catkin_ws/
    
compile:

    rm -r build/
    rm -r devel/
    catkin_make
    
source setup file:

    source ./devel/setup.bash
    
    
launch main script:

    roslaunch visual_multi_crop_row_navigation vs_navigation.py
    

### Dependencies:
- ROS Melodic
- python packages:

        sudo apt-get install python-catkin-tools python3-dev python3-catkin-pkg-modules python3-numpy python3-yaml ros-melodic-cv-bridge python3-opencv
        python3 -m pip install scikit-build scikit-learn laspy pandas
- build CV_Bridge for python3:

        cd catkin_ws
        git clone https://github.com/ros-perception/vision_opencv.git src/vision_opencv

    Find version:
    
        sudo apt-cache show ros-melodic-cv-bridge | grep Version
        Version: 1.12.8-0xenial-20180416-143935-0800
    
    Checkout right version in git repo. In our case it is 1.12.8

        cd src/vision_opencv/
        git checkout 1.12.8
        cd ../../

    build

        catkin_make --cmake-args \
            -DCMAKE_BUILD_TYPE=Release \
            -DPYTHON_EXECUTABLE=/usr/bin/python3 \
            -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
            -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
	 
## Citation 
if you use this project in your recent works please refernce to it by:

```bash

@article{ahmadi2021towards,
  title={Towards Autonomous Crop-Agnostic Visual Navigation in Arable Fields},
  author={Ahmadi, Alireza and Halstead, Michael and McCool, Chris},
  journal={arXiv preprint arXiv:2109.11936},
  year={2021}
}

@inproceedings{ahmadi2020visual,
  title={Visual servoing-based navigation for monitoring row-crop fields},
  author={Ahmadi, Alireza and Nardi, Lorenzo and Chebrolu, Nived and Stachniss, Cyrill},
  booktitle={2020 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={4920--4926},
  year={2020},
  organization={IEEE}
}
```

## Acknowledgments
This work has been supported by the German Research Foundation under Germany’s Excellence Strategy, EXC-2070 - 390732324 ([PhenoRob](http://www.phenorob.de/)) and [Bonn AgRobotics Group](http://agrobotics.uni-bonn.de/)
