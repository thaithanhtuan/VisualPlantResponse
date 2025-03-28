# VisualPlantResponse
This github is generate for the paper "Analyzing Plant Visual Behavior and Response Using a Low-Cost Time-Lapse System: A Case Study on Hydration Stress"
<div align="center">
  	<img src="Images/Point monitoring diagram.drawio.png" width="100%" />
	<p>Optical Flow-based plant motion tracking pipeline</p>
</div>

# Google drive link to Vitual Plant Response Phenomics Dataset
<div align="center">
  	<img src="Images/System design.png" width="576px" />
	<p>A controlled timelapse image acquisition system</p>
</div>

All Original data and generated data (Crop image, "angle, moving point, moving distance".csv, label.txt) are keep in this drive.
https://drive.google.com/drive/folders/1r1YIPgXkRS5ZeZN4X3-F8RF-05M9wM_U?usp=sharing

# Dependency
Package            Version
------------------ ---------
bokeh              3.1.0
brotlipy           0.7.0
certifi            2022.12.7
cffi               1.15.1
charset-normalizer 3.1.0
click              8.1.3
cloudpickle        2.2.1
colorama           0.4.6
contourpy          1.0.7
cryptography       40.0.2
cycler             0.11.0
cytoolz            0.12.0
dask               2023.4.0
dask-jobqueue      0.8.1
distributed        2023.4.0
et_xmlfile         2.0.0
FastLine           1.1
fonttools          4.39.3
fsspec             2023.4.0
idna               3.4
imagecodecs        2023.1.23
imageio            2.27.0
importlib-metadata 6.4.1
Jinja2             3.1.2
joblib             1.2.0
kiwisolver         1.4.4
lazy_loader        0.2
locket             1.0.0
lz4                4.3.2
MarkupSafe         2.1.2
matplotlib         3.5.3
mizani             0.9.0
msgpack            1.0.5
munkres            1.1.4
networkx           3.1
numpy              1.22.3
opencv-python      4.7.0
openpyxl           3.1.5
packaging          23.1
pandas             2.0.0
partd              1.4.0
patsy              0.5.3
Pillow             9.5.0
pip                23.1
plantcv            3.14.3
platformdirs       3.2.0
plotnine           0.10.1
pooch              1.7.0
psutil             5.9.5
pyarrow            11.0.0
pycparser          2.21
pyOpenSSL          23.1.1
pyparsing          3.0.9
PyQt5              5.15.9
PyQt5-Qt5          5.15.2
PyQt5-sip          12.12.2
PySocks            1.7.1
python-dateutil    2.8.2
pytz               2023.3
PyWavelets         1.4.1
PyYAML             6.0
pyzbar             0.1.9
requests           2.28.2
scikit-image       0.20.0
scikit-learn       1.2.2
scipy              1.10.1
setuptools         67.6.1
six                1.16.0
sortedcontainers   2.4.0
statsmodels        0.13.5
tblib              1.7.0
threadpoolctl      3.1.0
tifffile           2023.4.12
toolz              0.12.0
tornado            6.2
typing_extensions  4.5.0
tzdata             2023.3
unicodedata2       15.0.0
urllib3            1.26.15
wheel              0.40.0
win-inet-pton      1.1.0
XlsxWriter         3.1.2
xyzservices        2023.2.0
zict               3.0.0
zipp               3.15.0

# Output video
https://drive.google.com/file/d/1qUuuHTmpoioDFz2wFSou-HVYfeYRx0Bx/view?usp=sharing

# Output trajectories
<div align="center">
  <img src="Images/trajectories.jpg" width="576px" />
    <p>Trajectories of plant movement from above video.</p>
</div>

# How to run the code

## Combine time lapse images into videos (For Visualization Only)
Because the camera in Raspberry pi can place in vertical or horizontal direction, we need to set the rotation of the code, after changing to input directory and output video path.

```
image_folder = "Plant Optical Flow/Code/Out_dir"
output_video = "Plant Optical Flow/Code/Out_dir/timelapse_bg.mp4"
fps = 10  # Adjust frame rate as needed
```

Then run the code: 

```
python Make_video.py
```

## Run the interest Point Selection Tool
<div align="center">
  	<img src="Images/Point Selection tool.drawio.png" width="576px" />
	<p>Point selection tool flow diagram</p>
</div>

Changing the code and set:

first frame image

output label.txt file path

```
image_path = "VPRPDataset/Crop1/images/0000_cropped.jpg"  # Replace with your image path
output_file = "VPRPDataset/Crop1/label_F_L.txt"  # Replace with your desired output file name
```

Then run the script:

```
python PointSelectTool.py
```
While running: draw the bounding box using mouse to choose the interes points, press the number to choose the label, press 's' for saving reset label.txt and press 'q' for quit.

## Run the monitoring code
Set the image timelapse folder

Set the label.txt file

Set the output folder

```
image_folder = "VPRPDataset/Crop1/"
label_file = os.path.join(image_folder, "label_F_L.txt")
output_folder = "VPRPDataset/Crop1/output"
```

Then run the script:

```
python monitoringKeypoint.py
```
## To Calculate moving distance in 60 seconds

Set the input file and output file:

```
movingspeed_csv = "VPRPDataset/Crop2/output/movingspeed.csv"
movingspeed_60_csv = "VPRPDataset/Crop2/output/movingspeed_60.csv"
movingdistance_csv = "VPRPDataset/Crop2/output/movingdistance.csv"
input_distance_csv = "VPRPDataset/Crop2/output/movingdistance.csv"
output_distance_60_csv = "VPRPDataset/Crop2/output/movingdistance_60.csv"

```

```
python distance.py
```

## To Calculate Optical flow between two frames (for visualization)

```
python CalculateOF.py
```

# Citation
If this code helps your research, please cite our paper:

	@inproceedings{thaiplantphenomics,
		title={Analyzing Plant Visual Behavior and Response Using a Low-Cost Time-Lapse System: A Case Study on Hydration Stress},
		author={Thanh Tuan Thai and Jeong-Ho Baek and Sheikh Mansoor and E. M. B. M. Karunathilake and Anh Tuan Le and Sulieman Al-Faifi and Faheem Shehzad Baloch and Jinhyun Ahn and Yong Suk Chung},
		booktitle={},
		pages={},
		year={}
	}
# License
VisualPlant Response is freely available for free non-commercial use, and may be redistributed under these conditions. Please, see the [license](./LICENSE) for further details. For commercial queries, please contact [Prof.Yong Suk Chung](mailto:yschung@jejunu.ac.kr).
