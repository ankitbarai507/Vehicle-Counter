# Vehicle-Counter
**Basic philosophy** <br/>
We run object detection & classification model like Mobilenet,FRCNN(more optimal) to get current bounding boxes for each vehicles, then we track it using OpenCV Trackers to get updated bounding box coordinates and run the classification model again for next frames.
Now, we calculate IoU for the two bounding boxes and if IoU<threshlod then Vehicle count is incremented else it was a vehicle counted in previous frames.<br/>
Real time Vehicle counter using OpenCV trackers and FRCNN Object detection <br>
Download [frozen_inference_graph_frcnn.pb](https://github.com/mhBahrami/CarND-Capstone/blob/master/ros/src/tl_detector/models/sim/frozen_inference_graph_frcnn.pb)

**Testing**
```python
python track.py
```

**Results **
[Results1](https://youtu.be/rsgLz582Mfw)   <br/>
[Results2](https://youtu.be/8jpjXcMgrDM) 

---

A Vehicle Detection involves finding whether there is vehicle present or not secondly which type of vehicle is present and how many vehicles are present. Basically, vehicle presence needs to be detected after detecting a vehicle it has to be classified. Classification is the main part which means what type of vehicle it is (car, bus, bike, etc.). Smart Traffic Management aims at the avoidance of traffic on roads especially on highways because where manual governance and management is difficult. Implementation of traffic surveillance camera finds primary application in traffic monitoring through which management will become easier. If those cameras are enabled by the modern technology enables the Smart Traffic Management.

**YOLO (You Only look once ) Model**
It is a heavy architecture which is based on bounding boxes it cannot be used for embedded vision applications. Its trained on Pascal VOC, which can detect up to twenty different classes. Architecture of YOLO Yolo architecture is more like FCNN (fully constitutional neural network) and passes the image (nxn) once through the FCNN and output is (mxm) prediction. This the architecture is splitting the input image in mxm grid and for each grid generation 2 bounding boxes and class probabilities for those bounding boxes. Bounding box is more likely to be larger than the grid itself.
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")

**Limitations of YOLO**
+ YOLO imposes strong spatial constraints on bounding box predictions since each grid cell only predicts two boxes and can only have one class. This spatial constraint limits the number of nearby objects that our model can predict.
+ YOLO model struggles with small objects that appear in groups.
+ It treats errors the same in small bounding boxes versus large bounding boxes. A small error in a large box is generally benign but a small error in a small box has a much greater effect on IOU. Our main source of error is incorrect localizations.
+ YOLO is a heavy weight model 269.9MB, which is gives low recognition speed 2–3 fps and less accuracy.

