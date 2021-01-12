Robot reaction toward human according to vision data

Currently combined 3 method:
human pose and facial data detection: openpose - https://github.com/CMU-Perceptual-Computing-Lab/openpose
object tracking in video: motpy - https://github.com/wmuron/motpy
facial expression classifier: face_and_emotion_detection - https://github.com/priya-dwivedi/face_and_emotion_detection

Preparing Steps:
1. Install openpose (https://github.com/CMU-Perceptual-Computing-Lab/openpose).
	Win10:
	https://www.youtube.com/watch?v=QC9GTb6Wsb4 is a video for install openpose on win10
	https://www.programmersought.com/article/65535919191/ is the artical i used to setup my openpose under win10. Follow the instruction should be fine.
	
	Ubuntu:
	installation for Ubuntu is not completed yet, but openpose right now can work properly under 18.04lts, with my rtx2070. Here is a link to follow: https://alister-enikeev.medium.com/install-openpose-with-cuda-10-1-and-cudnn-for-ubuntu-18-04-in-2020-year-a9d559ae567e the link only work for older version of openpose, so there would be some modification:

2. Once you finish building up openpose, please download the folder “emotion_detector_models” (model comes from https://github.com/priya-dwivedi/face_and_emotion_detection
) and put the whole folder under the “build” folder (or whatever name you choose when create build directory through VS in the first step)

3. install motpy (https://github.com/wmuron/motpy)

4. Download the file demo1.py and put this file under the “build” folder

After all of steps above, simply run the demo1.py through python to see the result. It will directly use the camera connected to PC.

Warning: a Nvidia GPU is needed and a camera is needed.
