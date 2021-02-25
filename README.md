# SFU MSE ML Course Labs and Tools

## Labs

### [Lab 1](LAB_1)
### [Lab 2](LAB_2)

## Raspberry Pi Communication GUI and Raspberry Pi Script

### [Technical Manual](Ui_project/manual.md)
### [Troubleshooting Tips](Ui_project/manual.md#troubleshooting)
### [GUI Code](Ui_project)
### [Raspberry Pi Script](RPi_Script)

## Notes
* Pickle files that contain serialized scikit-learn (sklearn) models are not guaranteed to be compatible on different architectures, according to [this](https://scikit-learn.org/stable/modules/model_persistence.html). As the RPi used here has a 32-bit architecture, it is important that the sklearn models are trained and dumped to pickle files using a 32-bit python even if a 64-bit windows machine is used. To do that you can create a new conda environment for that using `conda create -n ml_course_32` then activate it `conda activate ml_course_32` then set the environment to accept 32 bit packages only using `conda config --env --set subdir win-32` and then install python using `conda install python=3.7` and then install the required packages for each lab as needed including `conda install scikit-learn`
