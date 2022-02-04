# Satellite Structure from Motion

Maintained by [Kai Zhang](https://kai-46.github.io/website/). 

## Overview
- This is a library dedicated to solving the satellite structure from motion problem.
- It's a wrapper of the [VisSatSatelliteStereo repo](https://github.com/Kai-46/VisSatSatelliteStereo) for easier use.
- The outputs are png images and OpenCV-compatible pinhole camreas readily deployable to multi-view stereo pipelines targetting ground-level images.

## Installation
Assume you are on a Linux machine, and have conda installed. Then to install this library, simply by:
```bash
. ./env.sh
```

## Inputs
We assume the inputs to be a set of .tif images encoding the 3-channel uint8 RGB colors, and the metadata like RPC cameras. 
This data format is to align with the public satellite benchmark: [TRACK 3: MULTI-VIEW SEMANTIC STEREO](https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019).
Download one example data from this [google drive](https://drive.google.com/drive/folders/11UeurSa-dyfaRUIdUZFfNBAyd3jN7D46?usp=sharing); folder structure look like below:


```
- examples/inputs
    - images/
        - *.tif
        - *.tif
        - *.tif
        - ...
    - latlonalt_bbx.json
```
, where ```latlonalt_bbx.json``` specifies the bounding box for the site of interest in the global (latitude, longitude, altitude) coordinate system.

## Run Structure from Motion
```bash
python satellite_sfm.py --input_folder examples/inputs --output_folder examples/outputs --run_sfm [--enable_debug]
```
The ```--enable_debug``` option outputs some visualization helpful debugging the structure from motion quality.

## Outputs
- ```{output_folder}/images/``` folder contains the png images
- ```{output_folder}/cameras_adjusted/``` folder contains the bundle-adjusted pinhole cameras
- ```enu_bbx_adjusted.json``` contains the scene bounding box in the local ENU Euclidean coordinate system.
- ```enu_observer_latlonalt.json``` contains the observer coordinate for defining the local ENU coordinate; essentially, this observer coordinate are necessary for coordinate conversion between local ENU and global latitude-longitude-altitude.
-
## Citations
```
@inproceedings{VisSat-2019,
 title={Leveraging Vision Reconstruction Pipelines for Satellite Imagery},
  author={Zhang, Kai and Sun, Jin and Snavely, Noah},
   booktitle={IEEE International Conference on Computer Vision Workshops},
    year={2019}
}
```
