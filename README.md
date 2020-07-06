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
Change the line no.362 in track.py with the video path(Video on which to test vehicle counter model)

**Results **
[Results1](https://youtu.be/rsgLz582Mfw)   <br/>
[Results2](https://youtu.be/8jpjXcMgrDM) 

---

A Vehicle Detection involves finding whether there is vehicle present or not secondly which type of vehicle is present and how many vehicles are present. Basically, vehicle presence needs to be detected after detecting a vehicle it has to be classified. Classification is the main part which means what type of vehicle it is (car, bus, bike, etc.). Smart Traffic Management aims at the avoidance of traffic on roads especially on highways because where manual governance and management is difficult. Implementation of traffic surveillance camera finds primary application in traffic monitoring through which management will become easier. If those cameras are enabled by the modern technology enables the Smart Traffic Management.

**YOLO (You Only look once ) Model**
It is a heavy architecture which is based on bounding boxes it cannot be used for embedded vision applications. Its trained on Pascal VOC, which can detect up to twenty different classes. Architecture of YOLO Yolo architecture is more like FCNN (fully constitutional neural network) and passes the image (nxn) once through the FCNN and output is (mxm) prediction. This the architecture is splitting the input image in mxm grid and for each grid generation 2 bounding boxes and class probabilities for those bounding boxes. Bounding box is more likely to be larger than the grid itself.


**Limitations of YOLO**
+ YOLO imposes strong spatial constraints on bounding box predictions since each grid cell only predicts two boxes and can only have one class. This spatial constraint limits the number of nearby objects that our model can predict.
+ YOLO model struggles with small objects that appear in groups.
+ It treats errors the same in small bounding boxes versus large bounding boxes. A small error in a large box is generally benign but a small error in a small box has a much greater effect on IOU. Our main source of error is incorrect localizations.
+ YOLO is a heavy weight model 269.9MB, which is gives low recognition speed 2–3 fps and less accuracy.

**Mobile Net-SSD Model Framework**
Mobile Nets SSD (Single Shot Multibox Detection) is an Efficient convolution Neural Network architecture for mobile and embedded vision applications. MobileNets are based on a streamlined architecture that uses depth wise separable convolutions to build light weight deep neural networks. Mobile Net is an architecture which is more suitable for mobile and embedded based vision applications where there is lack of compute power. This architecture was proposed by Google. This architecture uses depth wise separable convolutions which significantly reduces the number of parameters when compared to the network with normal convolutions with the same depth in the networks. This results in light weight deep neural networks. The normal convolution is replaced by depth wise convolution followed by point wise convolution which is called as depth wise separable convolution. By using depth wise separable convolutions, there is some sacrifice of accuracy for low complexity deep neural network. Employing Single Shot Multi-Box Detection compensate that and improves accuracy as well.
+ Single Shot: this means that the tasks of object localization and classification are done in a single forward pass of the network
+ Multi-box: this is the name of a technique for bounding box regression developed by Szegedy et al.
+ Detector: The network is an object detector that also classifies those detected objects.



