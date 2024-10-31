TELE-OPERATION

## Overview

Hello world =))

## Prerequisites

To run this repository, ensure the following requirements are met:

1. **Global Environment**
   - Define the global environment variable `MOCAP_WORKDIR` as the root folder of the entire project.

2. **Camera Calibration Files**
   - Calibrated files for two cameras are required. Set the paths to these files in the `camera` section of `./arm_and_hand/configuration/main_conf.yaml`.

3. **MMDeploy Library**
   - Since we use `mmdeploy` to run the deep learning model, confirm that this library is installed and properly configured.

### Software Specifications

This project is built on:

- **CUDA**: 11.8
- **TensorRT**: 8.6.1.6
- **PyTorch**: 2.1.2
- **MMCV**: 2.1.0

