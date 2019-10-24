# 360-panorama-python
Stitching 360 panorama with Python

This project generates 360 x 180 panorama from image pieces taken by cell phone cameras. The input data should contain image meta data in JSON format. 
This meta data is consisted of camera metrics data and image meta data.

* Camera metrics: focal length and sensor size which can be retrieved by Android API.
* Image metadata: RPY(Roll, Pitch, and Yaw) data for every image

This project contains `.py` files which generates panorama from image pieces. Some files are in Jupyter (IPython) notebooks (exp-*.ipynb), which describes experiments to develop this projects. You can read here how the stitching pipeline works and what ideas this project is based on.

## Requirements
- Python >= 3.x
- opencv-python >= 3.4
- numpy >= 1.0.0
- scipy >= 1.0.0
- matplotlib >= 2.0
- imutils >= 0.5.2
- pandas >= 0.24.2

Python and opencv-python are highly dependent on versions but others doesn't need strict version requirements.

## Installation

Download the project and run following commands:

### Linux
```bash
sudo bash ./install_requirements.sh
```

## Usage

```bash
python3 ./start.py {image folder} [--width WIDTH] [--height HEIGHT] [--output OUTPUT]
```
- {image folder}: the path of the folder containing images and the meta JSON file.
- WIDTH: desired panorama width in pixels (default 4096)
- HEIGHT: desired panorama height in pixels (default 2048)
- OUTPUT: file name to store the panorama file (default panorama.jpg)
