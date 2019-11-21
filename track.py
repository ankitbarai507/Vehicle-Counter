import cv2
import numpy as np

# params to tune for a given resolution
out_limit = 15
in_limit = 25
interval = 15


class Tracker(object):
    def __init__(self, tracker, idno):
        self.tracker = tracker
        self.idno = idno

    def __del__(self):
        print("Deleting Tracker No.: ", self.idno)


classes_90 = ["background", "person", "bicycle", "car", "motorcycle",
              "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
              "unknown", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
              "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "unknown", "backpack",
              "umbrella", "unknown", "unknown", "handbag", "tie", "suitcase", "frisbee", "skis",
              "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
              "surfboard", "tennis racket", "bottle", "unknown", "wine glass", "cup", "fork", "knife",
              "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
              "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "unknown", "dining table",
              "unknown", "unknown", "toilet", "unknown", "tv", "laptop", "mouse", "remote", "keyboard",
              "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "unknown",
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


def convert_to_dict(bboxes):
    """
     converts a list of bounding boxes(x1,y1,x2,y2) into a dictionary
    :param bboxes: A list of bounding boxes, each of which is a tuple (x1,y1,x2,y2)
    :return: A list of bounding boxes in dict format
    """
    dict_list = []
    for bbox in bboxes:
        d = dict()
        d['x1'] = bbox[0]
        d['y1'] = bbox[1]
        d['x2'] = bbox[2]
        d['y2'] = bbox[3]
        if not out_of_frame(d, frame, -in_limit):
            dict_list.append(d)
    return dict_list


def get_dict(bbox):
    """
    Converts BBOX (x,y,w,h) =>{x1:,y1:,x2:,y2:}
    :param bbox: A tuple (x,y,w,h)
    :return: Dictionary: {x1:,y1:,x2:,y2:}
    """
    d = dict()
    d['x1'] = bbox[0]
    d['x2'] = bbox[0] + bbox[2]
    d['y1'] = bbox[1]
    d['y2'] = bbox[1] + bbox[3]
    return d


def non_max_suppression_fast(boxes, overlap_thresh):
    """
    Performs Vectorized Non Max suppression on the given list of bounding boxes.
    :param boxes: A list of bounding boxes, (each bounding box is a dictionary)
    :param overlap_thresh: The threshold IOU used to tell if two BBoxes are on the same object.
    :return: A list of Non-Max-Suppressed BBoxes
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_thresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return convert_to_dict(boxes[pick].astype("int"))


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    :param bb1: Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    :param bb2: Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    :type bb1: dict
    :type bb2: dict
    :return: float
         in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    if (bb1['x1'] > bb2['x1'] and bb1['x2'] < bb2['x2']) or (bb1['x1'] < bb2['x1'] and bb1['x2'] > bb2['x2']):
        return 1
    elif (bb1['y1'] > bb2['y1'] and bb1['y2'] < bb2['y2']) or (bb1['y1'] < bb2['y1'] and bb1['y2'] > bb2['y2']):
        return 1
    return iou


def get_tracker(tracker_type):
    """
    Returns an OpenCV tracker.
    :param tracker_type: A string denoting the type of tracker to return
    :type tracker_type: str
    :return: OpenCV tracker.
    """

    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    return tracker


def get_dnn_boxes(tensorflowNet, frame):
    """
    takes the model, performs forward pass on the given frame, and returns a list of NMS'ed BBoxes
    :param tensorflowNet: a cv2 DNN object using which forward pass will be performed
    :param frame: The image(frame) on which inference is to be performed.
    :return: Bounding boxes for relevant objects, after performing NMS.
    """

    tensorflowNet.setInput(cv2.dnn.blobFromImage(
        frame, size=(300, 300), swapRB=True, crop=False))
    network_output = tensorflowNet.forward()
    rows, cols, _ = frame.shape
    bboxes = []
    # Loop on the outputs
    for detection in network_output[0, 0]:
        score = float(detection[2])
        if score > 0.35:
            detected = classes_90[int(detection[1])]
            if detected in ["bus", "train", "truck", "bicycle", "car", "motorcycle"]:
                newbox = dict()
                newbox['x1'] = detection[3] * cols
                newbox['y1'] = detection[4] * rows
                newbox['x2'] = detection[5] * cols
                newbox['y2'] = detection[6] * rows
                bboxes.append([newbox['x1'], newbox['y1'],
                               newbox['x2'], newbox['y2']])
    return non_max_suppression_fast(np.asarray(bboxes), 0.5)


def out_of_frame(bbox, frame, limit=0):
    """
    Returns True if the bounding box is out of the frame
    :param bbox: A dict having the coordinates of the bounding box.
    :param frame: A cv2 image in which the bounding box is present.
    :param limit: An integer, which accounts for the error in the bounding box, can be negative.
    :return: True if the bounding box is out of the frame,else False
    """

    h, w, _ = frame.shape
    if (bbox['x1'] < 0 - limit) or (bbox['y1'] < 0 - limit) or (bbox['x2'] > w + limit) or (bbox['y2'] > h + limit):
        return True
    return False


tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD',
                 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[-1]
tensorflowNet = cv2.dnn.readNetFromTensorflow(
    'frozen_inference_graph_frcnn.pb', 'pbpb_frcnn.pbtxt')

frame = None


# Read video


def main(video_path):
    """
    Performs car counting on the given video file
    :param video_path: string, the absolute path of the video, in which cars are to be counted.
    :return: int, Number of cars detected in the video.
    """
    video = cv2.VideoCapture(video_path)
    global frame
    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        return -1
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
    frame = cv2.resize(frame, (1280, 720))
    out = cv2.VideoWriter('outpy' + video_path.replace("/", "_") + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30,
                          (frame.shape[1], frame.shape[0]))

    bboxes = get_dnn_boxes(tensorflowNet, frame)
    tracker_list = []
    car_count = 0
    frame_count = 1

    # Initialize tracker with first frame and bounding box
    for bbox in bboxes:
        car_count += 1
        tracker_box = (bbox['x1'], bbox['y1'], bbox['x2'] -
                      bbox['x1'], bbox['y2'] - bbox['y1'])
        tracker = get_tracker(tracker_type)
        ok = tracker.init(frame, tracker_box)
        tracker = Tracker(tracker, car_count)
        tracker_list.append(tracker)

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            print("breaking from inside the loop")
            break
        frame = cv2.resize(frame, (1280, 720))  # scale it down from 4k

        # update the current trackers, remove trackers that are done
        tlist1 = []
        bbox_list = []
        for tracker in tracker_list:
            ok, bbox = tracker.tracker.update(frame)
            bbox = get_dict(bbox)
            ok = False if out_of_frame(bbox, frame, out_limit) else True
            if ok:
                tlist1.append(tracker)
                bbox_list.append(bbox)
                cv2.putText(frame, str(tracker.idno), (int(bbox['x1']), int(bbox['y1']) - 15), 0, 0.8, (0, 0, 255), 2,
                            cv2.LINE_AA)
                cv2.rectangle(frame, (int(bbox['x1']), int(bbox['y1'])), (int(bbox['x2']), int(bbox['y2'])),
                              (0, 0, 255),
                              thickness=5)
            else:
                print("Deleting tracker", tracker.idno)

        tracker_list = tlist1

        if frame_count % interval == 0:
            # use detection
            try:
                bboxes = get_dnn_boxes(tensorflowNet, frame)
            except Exception as e:
                frame_count -= 1  # try to infer again next frame
                print(e)
            for bbox in bboxes:
                flag = True
                for bbox1 in bbox_list:
                    if get_iou(bbox, bbox1) >= 0.3:  # already being tracked
                        flag = False
                if flag:
                    car_count += 1
                    tracker_box = (
                        bbox['x1'], bbox['y1'], bbox['x2'] - bbox['x1'], bbox['y2'] - bbox['y1'])
                    tracker = get_tracker(tracker_type)
                    ok = tracker.init(frame, tracker_box)
                    tracker_list.append(Tracker(tracker, car_count))

        # Draw bounding box
        frame_count += 1
        cv2.putText(frame, "No of cars : " + str(car_count),
                    (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (50, 170, 50), 2)
        # Display result
        out.write(frame)
        cv2.imshow("Tracking", frame)
        print(car_count)
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
    out.release()
    print("Done , count is ", car_count)
    cv2.destroyAllWindows()
    return car_count


if __name__ == "__main__":
    print(main("los_angeles.mp4"))
