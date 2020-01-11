# Vehicle-Counter
**Basic philosophy** <br/>
We run object detection & classification model like Mobilenet,FRCNN(more optimal) to get current bounding boxes for each vehicles, then we track it using OpenCV Trackers to get updated bounding box coordinates and run the classification model again for next frames.
Now, we calculate IoU for the two bounding boxes and if IoU<threshlod then Vehicle count is incremented else it was a vehicle counted in previous frames.<br/>
Real time Vehicle counter using OpenCV trackers and FRCNN Object detection <br>
Download [frozen_inference_graph_frcnn.pb](https://github.com/mhBahrami/CarND-Capstone/blob/master/ros/src/tl_detector/models/sim/frozen_inference_graph_frcnn.pb)

Change video file name in main function
Run python track.py <br>

[Results1](https://youtu.be/rsgLz582Mfw)  <br/>
[Results2](https://youtu.be/8jpjXcMgrDM)
