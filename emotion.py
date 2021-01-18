from __future__ import print_function

"""
Some useful API for you!
"""

import qi
import sys
import time
import random
import motion
import matplotlib.pyplot as plt
import numpy as np
import cv2

# ==============================================================================
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2

from imutils.object_detection import non_max_suppression
from imutils import paths
import imutils

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==============================================================================


# Create the model for pose
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(100, 250, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# Create the model for emotion
model2 = Sequential()

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))

model2.add(Flatten())
model2.add(Dense(1024, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(7, activation='softmax'))

# ==============================================================================
#                      --- CAMERA INFORMATION ---
#                          No need to change
# AL_resolution
AL_kQQQQVGA = 8  # Image of 40*30px
AL_kQQQVGA = 7  # Image of 80*60px
AL_kQQVGA = 0  # Image of 160*120px
AL_kQVGA = 1  # Image of 320*240px
AL_kVGA = 2  # Image of 640*480px
AL_k4VGA = 3  # Image of 1280*960px
AL_k16VGA = 4  # Image of 2560*1920px

# Camera IDs
AL_kTopCamera = 0
AL_kBottomCamera = 1
AL_kDepthCamera = 2

# Need to add All color space variables
AL_kBGRColorSpace = 13

# ==============================================================================

# connect the pepper
session = qi.Session()
ip = "***.**.**.*" # change according to your own pepper setting
port = **** # change according to your own pepper setting
session.connect("tcp://" + ip + ":" + str(port))

# we need to register some services to control the robot
motion_service = session.service("ALMotion")
posture_service = session.service("ALRobotPosture")
speech_service = session.service("ALTextToSpeech")
animation_player_service = session.service("ALAnimationPlayer")
video_service = session.service("ALVideoDevice")
face_detection_service = session.service("ALFaceDetection")  # newly added
memory_service = session.service("ALMemory")  # newly added
photo_capture_service = session.service("ALPhotoCapture")  # newly added

####### initialization #########
motion_service.wakeUp()
posture_service.goToPosture("StandInit", 0.5)
motion_service.setMoveArmsEnabled(True, True)
motion_service.setMotionConfig([["ENABLE_FOOT_CONTACT_PROTECTION", True]])

#######  speak example  ##########

# text = "hello, world"
# speech_service.say(text)


#######  Attract Attention  ##########

text = "Hi,I am Pepper. Would you like to share with me your mood at the moment? Please allow me to see your pretty face."
speech_service.say(text)


#######  Take Pictures ##########
def decodeImage(data):
    if data:
        imageWidth = data[0]
        imageHeight = data[1]
        imageLayers = data[2]
        imageArray = data[6]
        imageArray = np.array(bytearray(imageArray))
        im = imageArray.reshape([imageHeight, imageWidth, imageLayers])
        return np.array(im)
    return None


from naoqi import ALProxy

vision = ALProxy("ALVideoDevice", ip, port)
while 1:
    time.sleep(3)
    ImageData = vision.getImageRemote("FrameGetter/Camera_1")
    Image = decodeImage(ImageData)
    # cv2.imshow("Image",Image)
    break
    k = cv2.waitKey(1)
    if k == ord("q"):
        break

# cv2.imshow('Image', cv2.resize(Image,(1600,960),interpolation = cv2.INTER_CUBIC))

#######  Download Pictures from Pepper ##########

# t = paramiko.Transport(("172.20.10.6", 9559))
# t.connect(username="nao", password="51749110")
# sftp = paramiko.SFTPClient.from_transport(t)


# files=sftp.listdir(image_path)
# for f in files:
#    print ('')
#    print ('##############################################')
#    print ('Beginning to download file from %s %s' %("172.20.10.6",datetime.datetime.now()))
#    print ('Downloading file:', os.path.join(image_path,f))


# sftp.get(os.path.join(image_path,f),os.path.join(local_path,f))
# print ('Download file success %s'%datetime.datetime.now())
# print ('')
# print ('###############################################')
# t.close()


#######  Predicting the Pictures Implementation ##########

model.load_weights('model.h5')
model2.load_weights('model2.h5')

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
pose_dict = {0: "Active", 1: "Inactive"}
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
print("[INFO] running pedestrian detection...")

(rects, _) = hog.detectMultiScale(Image, winStride=(4, 4), padding=(8, 8), scale=1.05)

gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)

# Applies non-max supression from imutils package
# to kick-off overlapped boxes
pose_predicted = -1
emotion_predicted = -1
pose_index = -1

rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
result = non_max_suppression(rects, probs=None, overlapThresh=0.05)
# draw the final bounding boxes
for (xA, yA, xB, yB) in result:
    cv2.rectangle(Image, (xA, yA), (xB, yB), (0, 255, 0), 2)
    roi_gray = gray[yA:yA + yB, xA:xA + xB]
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (250, 100)), -1), 0)
    prediction = model.predict(cropped_img)
    maxindex = int(np.argmax(prediction))
    pose_predicted = pose_dict[maxindex]

facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# imgPath='/home/xuweirosa/Emotion-detection/src/1.jpg'
gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

for (x, y, w, h) in faces:
    cv2.rectangle(Image, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
    prediction = model2.predict(cropped_img)
    if (pose_index == 0):
        prediction[0] = prediction[0] * 1.5
        prediction[3] = prediction[3] * 1.5
        prediction[6] = prediction[6] * 1.5
    if (pose_dict == 1):
        prediction[1] = prediction[1] * 1.5
        prediction[2] = prediction[2] * 1.5
        prediction[4] = prediction[4] * 1.5
        prediction[5] = prediction[5] * 1.5
    maxindex = int(np.argmax(prediction))
    emotion_predicted = emotion_dict[maxindex]

#######  Reaction  ##########

names, keys, times = act.think()
motion_service.angleInterpolation(names, keys, times, True)

if (emotion_predicted == -1):
    text = "Sorry, I can not see your face. Maybe we can play this game next time."
    speech_service.say(text)
else:
    text = "Oh, it seems that you are " + emotion_predicted
    speech_service.say(text)







# expression
##########################################################################


import csv
import threading
import act as act


class EmotionRobot(object):
    def __init__(self, session):
        self.session = session
        self.module_name = "Emotion"
        self.emotion = emotion_predicted
        self.file_writer = csv.writer(open('test.csv', 'wb'), delimiter=',')
        self.size = 0.7

    def start(self):
        self.subscribingLock = threading.Lock()
        self.setALMemorySubscription(True)
        # print self.module_name, "service started..."

    def setALMemorySubscription(self, subscribe):
        self.subscribingLock.acquire()
        if subscribe:
            self.on_user_values()
        self.subscribingLock.release()

    def on_user_values(self):  # calculate activated emotion robot with user values
        # print 'on user values'

        #if self.emotion is not None:
        #    self.file_writer.writerow(
        #        ["recognized face emotion: ", self.emotion, "\n"])
        #else:
        #    print ('no face recognized')

        self.on_activated_emotion(self.emotion)

    def on_activated_emotion(self, emotion):  # raise events to express emotion

        # express emotion
        self.set_led(emotion)
        self.set_speech_style(emotion)
        self.set_speech_rate(emotion)
        self.set_voice_pitch(emotion)
        self.set_speech_volume(emotion)
        self.set_gesture_size(emotion)
        self.set_position(emotion)
        self.set_action1(emotion)
        self.set_speech(emotion)
        self.set_action2(emotion)
        speech_service.say("end")

    def set_speech(self, emotion):
        text = ""
        if emotion == "Sad":
            text = "It's hard for me to see you sad. Let me cheer you up. Guess what the animal I'm imitating."
        # elif emotion == "Happy":
        # text = "You are most charming when you are happy~ I am so happy to see you so happy!"
        elif emotion == "Disgusted":
            text = "What's up? I've always been by your side."
        elif emotion == "Angry":
            text = "Life never goes we planed. Everything will be fine! Let me sing for you and forget about the unhappiness~"
        elif emotion == "Surprised":
            text = "You scared me."
        elif emotion == "Fearful":
            text = "Don't be afraid. I'll protect you."
        elif emotion == "Neutral":
            text = "Pepper likes you the most! I'll always be with you~"

        self.file_writer.writerow(["text ", text, "\n"])
        # print 'text' + text
        speech_service.say(text)

    def set_led(self, emotion):
        color = None
        if emotion == "Sad":  # purple
            color = 0x006739b6
        elif emotion == "Happy":
            chance = random.randint(0, 1)
            if chance == 1:
                color = 0x00ffb400      # yellow
                self.leds().fadeRGB("AllLeds", color, 0.1)
                text = "You are most charming when you are happy~ I am so happy to see you so happy!"
            else:
                color = "red"
                self.leds().fadeRGB("AllLeds", color, 0.1)
                text = "I'm angry because you're happy. I'm just kidding. hahaha. Wish you happy every day."

            speech_service.say(text)
        elif emotion == "Disgusted":  # green/yellow
            color = 0x008dc63f
        elif emotion == "Angry":  # blue
            color = "blue"
        elif emotion == "Surprised":  # orange
            color = 0x00ff8d00
        elif emotion == "Fearful":  # green
            color = 0x0039b54a
        elif emotion == "Neutral":  # pink
            color = 0x00e03997

        self.file_writer.writerow(["color ", str(color), "\n"])
        # print 'color' + str(color)
        if emotion == "Surprised":
            self.leds().fadeRGB("AllLeds", color, 0.05)
            self.leds().fadeRGB("AllLeds", 0x000000, 0.05)
            self.leds().fadeRGB("AllLeds", color, 0.05)
            self.leds().fadeRGB("AllLeds", 0x000000, 0.05)
            self.leds().fadeRGB("AllLeds", color, 0.05)
        else:
            self.leds().fadeRGB("AllLeds", color, 0.1)

    def set_speech_style(self, emotion):
        style = None
        if emotion == "Sad":
            style = "neutral"
        elif emotion == "Happy":
            style = "joyful"
        elif emotion == "Disgusted":
            style = "neutral"
        elif emotion == "Angry":
            style = "neutral"
        elif emotion == "Surprised":
            style = "joyful"
        elif emotion == "Fearful":
            style = "neutral"
        elif emotion == "Neutral":
            style = "joyful"

        self.file_writer.writerow(["style ", style, "\n"])
        # print 'style' + style
        self.mem().insertData("Emotion/style", style)

    def set_speech_rate(self, emotion):
        rate = None
        if emotion == "Sad":
            rate = 75
        elif emotion == "Happy":
            rate = 90
        elif emotion == "Disgusted":
            rate = 85
        elif emotion == "Angry":
            rate = 85
        elif emotion == "Surprised":
            rate = 95
        elif emotion == "Fearful":
            rate = 85
        elif emotion == "Neutral":
            rate = 90

        self.file_writer.writerow(["rate ", str(rate), "\n"])
        # print 'rate' + str(rate)
        self.tts().setParameter("speed", rate)

    def set_voice_pitch(self, emotion):
        pitch = None
        if emotion == "Sad":
            pitch = 90
        elif emotion == "Happy":
            pitch = 110
        elif emotion == "Disgusted":
            pitch = 100
        elif emotion == "Angry":
            pitch = 100
        elif emotion == "Surprised":
            pitch = 120
        elif emotion == "Fearful":
            pitch = 100
        elif emotion == "Neutral":
            pitch = 110

        self.file_writer.writerow(["pitch ", str(pitch), "\n"])
        # print 'pitch' + str(pitch)
        self.mem().insertData("Emotion/pitch", pitch)

    def set_speech_volume(self, emotion):
        volume = None
        default_volume = self.tts().getVolume()
        if emotion == "Sad":
            volume = default_volume * 0.9
        elif emotion == "Happy":
            volume = default_volume * 1.1
        elif emotion == "Disgusted":
            volume = default_volume
        elif emotion == "Angry":
            volume = default_volume
        elif emotion == "Surprised":
            volume = default_volume * 1.2
        elif emotion == "Fearful":
            volume = default_volume
        elif emotion == "Neutral":
            volume = default_volume * 1.1

        self.file_writer.writerow(["volume ", str(volume), "\n"])
        # print 'volume' + str(volume)
        self.mem().insertData("Emotion/volume", volume)

    def set_gesture_size(self, emotion):
        size = self.size
        if emotion == "Sad":
            size = 0.6
        elif emotion == "Happy":
            size = 0.9
        elif emotion == "Disgusted":
            size = 0.75
        elif emotion == "Angry":
            size = 0.75
        elif emotion == "Surprised":
            size = 1.0
        elif emotion == "Fearful":
            size = 0.75
        elif emotion == "Neutral":
            size = 0.9

        self.size = size
        self.file_writer.writerow(["gesture size ", str(size), "\n"])
        # print 'gesture size' + str(size)
        names = "LArm"
        stiffnessLists = size
        timeLists = 1.0
        self.ms().stiffnessInterpolation(names, stiffnessLists, timeLists)
        names = "RArm"
        stiffnessLists = size
        timeLists = 1.0
        self.ms().stiffnessInterpolation(names, stiffnessLists, timeLists)

    def set_action1(self, emotion):

        if emotion == "Disgusted":  # curious
            names, keys, times = act.curious()
            motion_service.angleInterpolation(names, keys, times, True)

        elif emotion == "Surprised":
            names = "HeadYaw"
            set_angles = -0.3
            self.ms().setAngles(names, set_angles, 0.2)

            JointNames = ["LElbowRoll", "LElbowYaw", "LHand", "LShoulderPitch", "LShoulderRoll", "LWristYaw",
                          "RElbowRoll", "RElbowYaw", "RHand", "RShoulderPitch", "RShoulderRoll", "RWristYaw"]
            # # joints angle state 1
            Arm1 = [-89.4, -59.6, 0.98, 20.8, 7.7, -104.4,
                    89.3, 58.7, 0.98, 20.4, -7.2, 104.4]
            Arm1 = [x * motion.TO_RAD for x in Arm1]
            Arm2 = [-89.3, -51.9, 0.71, 10.6, 2.2, -64.7,
                    86.0, 44.3, 0.76, 2.4, -13.8, 57.9]
            Arm2 = [x * motion.TO_RAD for x in Arm2]  # kiss
            pFractionMaxSpeed = 0.2
            # # change the angle of joints
            # motion_service.angleInterpolationWithSpeed(JointNames, Arm1, pFractionMaxSpeed)
            motion_service.angleInterpolationWithSpeed(JointNames, Arm2, pFractionMaxSpeed)
            speech_service.say("Oh!~")
            time.sleep(1.0)
            # # reset
            posture_service.goToPosture("StandInit", 0.5)


    def set_action2(self, emotion):


        if emotion == "Sad":
            chance = random.randint(0, 1)
            if chance == 1:
                names, keys, times = act.animal()
                motion_service.angleInterpolation(names, keys, times, True)
                speech_service.say("It's a gorilla. Ha ha ha.")
            else:
                names, keys, times = act.animal2()
                motion_service.angleInterpolation(names, keys, times, True)
                speech_service.say("It's an elephant. Ha ha ha.")
        elif emotion == "Happy":
            names, keys, times = act.happy()
            motion_service.angleInterpolation(names, keys, times, True)

        elif emotion == "Angry":
            player = session.service("ALAudioPlayer")

            # audiofile = "1.mp3"
            # from os import path
            # AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), audiofile)
            fileId = "/home/nao/music/1.mp3"
            player.playFile(fileId)


        elif emotion == "Neutral":
            names, keys, times = act.kiss()
            motion_service.angleInterpolation(names, keys, times, True)


    def set_position(self, emotion):
        names = "HeadPitch"
        useSensors = True
        current_angles = self.ms().getAngles(names, useSensors)
        set_angles = current_angles
        chance = random.randint(0, 2)

        if emotion == "Sad" and (current_angles < 0):  # if sad always look down
            set_angles = 0.3
        elif emotion == "Surprised" and (current_angles > 0):  # if excited always look up
            set_angles = -0.3
        elif (current_angles < 0) and (chance == 0):  # looking up, then chance 33% to look down
            set_angles = 0.2
        elif (current_angles > 0):  # looking down, then look up
            set_angles = -0.2

        self.file_writer.writerow(["head angle ", str(set_angles), "\n"])
        # print 'head angle' + str(set_angles)
        self.ms().setAngles(names, set_angles, 0.2)

    def mem(self):
        return self.session.service("ALMemory")

    def leds(self):
        return self.session.service("ALLeds")

    def tts(self):
        return self.session.service("ALTextToSpeech")

    def ms(self):
        return self.session.service("ALMotion")


asr = session.service("ALSpeechRecognition")
# asr.setLanguage("Chinese")
# speech_service.setLanguage("Chinese")
emotion_robot = EmotionRobot(session)
emotion_robot.start()