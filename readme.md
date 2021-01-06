Robot reaction toward human according to vision data

Currently combined 3 method:
human pose and facial data detection: openpose - https://github.com/CMU-Perceptual-Computing-Lab/openpose
object tracking in video: motpy - https://github.com/wmuron/motpy
facial expression classifier: face_and_emotion_detection - https://github.com/priya-dwivedi/face_and_emotion_detection

Preparing Steps:
1. Install openpose (https://github.com/CMU-Perceptual-Computing-Lab/openpose).
	https://www.youtube.com/watch?v=QC9GTb6Wsb4 is the video i used to setup my openpose under win10. Follow the instruction in the video.

2. Once you finish building up openpose, please download the folder “emotion_detector_models” (model comes from https://github.com/priya-dwivedi/face_and_emotion_detection
) and put the whole folder under the “build” folder (or whatever name you choose when create build directory through VS in the first step)

3. install motpy (https://github.com/wmuron/motpy)

4. Download the file demo1.py and put this file under the “build” folder

After all of steps above, simply run the demo1.py through python to see the result. It will directly use the camera connected to PC.

Warning: a Nvidia GPU is needed and a camera is needed.
