# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time
from urllib.request import urlretrieve
from loguru import logger
from motpy import Detection, MultiObjectTracker
from motpy.testing_viz import draw_detection, draw_track
import numpy as np
font = cv2.FONT_HERSHEY_SIMPLEX

from keras.models import model_from_json
emotionclassifier = model_from_json(open("./emotion_detector_models/facial_expression_model_structure.json", "r").read())
emotionclassifier.load_weights('./emotion_detector_models/facial_expression_model_weights.h5')
emotionclass = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
# ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/x64/Release;' +  dir_path + '/bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('/python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="examples/media/COCO_val2014_000000000536.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "models/"
    params["face"] = True

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    
    faceRectangles = [
        op.Rectangle(330.119385, 277.532715, 48.717274, 48.717274),
        op.Rectangle(24.036991, 267.918793, 65.175171, 65.175171),
        op.Rectangle(151.803436, 32.477852, 108.295761, 108.295761),
    ]

    #tracking
    model_spec = {'order_pos': 1, 'dim_pos': 2,
                  'order_size': 0, 'dim_size': 2,
                  'q_var_pos': 5000., 'r_var_pos': 0.1}

    dt = 1 / 10  # assume 15 fps
    tracker = MultiObjectTracker(dt=dt, model_spec=model_spec)

    cap = cv2.VideoCapture(0)

    frames = 0
    while True:
        ret, frame = cap.read()
        # Read image and face rectangle locations
        imageToProcess = frame
        # Create new datum
        datum = op.Datum()
        datum.cvInputData = imageToProcess
        datum.faceRectangles = faceRectangles

        # Process and display image
        opWrapper.emplaceAndPop([datum])
        posedata = datum.poseKeypoints.copy()
        facedata = datum.faceKeypoints.copy()
        tmpp = datum.cvInputData.copy()
        bboxes = []
        for i in range(len(posedata)):
            # tmpp = cv2.circle(tmpp, (posedata[i][0][0], posedata[i][0][1]), 5, (255, 0, 0), 2)
            eye1 = posedata[i][15]
            eye2 = posedata[i][16]
            if eye1[2] > 0.5 and eye2[2] > 0.5:
                xmax = max(eye1[0], eye2[0])
                ymax = max(eye1[1], eye2[1])
                xmin = min(eye1[0], eye2[0])
                ymin = min(eye1[1], eye2[1])
                diff = xmax - xmin
                xmin = max(int(xmin - (diff*1)), 0)
                xmax = min(int(xmax + (diff*1)), frame.shape[1])
                ymin = max(int(ymin - (diff*2)), 0)
                ymax = min(int(ymax + (diff*2)), frame.shape[0])
                if xmax != xmin and ymax != ymin:
                    bboxes.append([xmin, ymin, xmax, ymax])
            # tmpp = cv2.circle(tmpp, (eye1[0], eye1[1]), 2, (0, 255, 0), 2)
            # tmpp = cv2.circle(tmpp, (eye2[0], eye2[1]), 2, (0, 255, 0), 2)
        if len(bboxes) > 0:
            detections = [Detection(box=bbox) for bbox in bboxes]
            logger.debug(f'detections: {detections}')

            tracker.step(detections)
            tracks = tracker.active_tracks(min_steps_alive=1)
            logger.debug(f'tracks: {tracks}')

            # preview the boxes on frame
            # for det in detections:
            #     draw_detection(tmpp, det)

            for facep in bboxes:
                facepic = frame[facep[1]: facep[3], facep[0]: facep[2]]
                grayf = cv2.cvtColor(facepic.copy(),cv2.COLOR_BGR2GRAY)
                roi_gray = cv2.resize(grayf, (48, 48), interpolation = cv2.INTER_AREA)
                roi_gray = np.expand_dims(roi_gray, axis=0)
                roi_gray = np.expand_dims(roi_gray, axis=3)
                predicted_class = np.argmax(emotionclassifier.predict(roi_gray)[0])
                cv2.putText(tmpp, emotionclass[predicted_class], (facep[0],int((facep[1] + facep[3])/2)), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            for track in tracks:
                draw_track(tmpp, track)
        frames = frames + 1
        cv2.imshow("OpenPose 1.6.0 - Video", tmpp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(e)
    sys.exit(-1)
