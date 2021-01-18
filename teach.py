import qi
import time
import threading


def playsound():
    aup.playFile("/home/nao/music/2.mp3")
    return


def doMouvement(mouvement):
    speech_service.say("I'm moving")

    for i in range(0, len(mouvement)):
        changes = mouvement[i]
        fractionMaxSpeed = 0.2
        motion_service.setAngles("Body", changes, fractionMaxSpeed)
        time.sleep(0.1)
    speech_service.say("I'm finish")


def getMouvement():
    stiffnessOff()
    speech_service.say("Touch my head when you're done")
    useSensors = True
    names = "Body"
    sensorAngles = []
    # print(int(touched()))
    while touched() == False:
        sensorAngles.append(motion_service.getAngles(names, useSensors))
        time.sleep(0.1)
    speech_service.say('end')
    # Save the dance files to excel
    file = open("dances.xls", "w")
    file.write(str(sensorAngles))
    file.close()
    # memory_service.insertData("Dance1", sensorAngles)
    stiffnessOn()
    return sensorAngles


def touched():      # touch the head
    # print (touch.getStatus()[9])
    return touch.getStatus()[9][1]


def stiffnessOff():
    stiffnames = ["Head", "RArm", "LArm", "Leg"]
    stiffnessLists = 0.0
    timeLists = 1.0
    for i in range(3):
        motion_service.stiffnessInterpolation(stiffnames[i], stiffnessLists, timeLists)


def stiffnessOn():
    stiffnames = ["Head", "RArm", "LArm", "Leg"]
    stiffnessLists = 1.0
    timeLists = 1.0
    for i in range(3):
        motion_service.stiffnessInterpolation(stiffnames[i], stiffnessLists, timeLists)


# connect the pepper
session = qi.Session()
ip = "172.20.10.6"
port = 9559
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
animatedSpeechProxy = session.service("ALAnimatedSpeech")
asr = session.service("ALSpeechRecognition")
aup = session.service("ALAudioPlayer")
touch = session.service("ALTouch")

####### initialization #########
motion_service.wakeUp()
posture_service.goToPosture("StandInit", 0.5)
motion_service.setMoveArmsEnabled(True, True)
motion_service.setMotionConfig([["ENABLE_FOOT_CONTACT_PROTECTION", True]])





configuration = {"bodyLanguageMode": "random"}
animatedSpeechProxy.say("Hi! I am Pepper! Yes, I am a real dancer!", configuration)

while True:
    dance = []
    # motion_service.wakeUp()
    # tmp = threading.Thread(target=playsound)
    # tmp.start()
    configuration = {"bodyLanguageMode": "random"}
    for i in range(3):
        if i == 0:      # teach
            dance = getMouvement()
        elif i == 1:        # dance
            # dance = memory_service.getData("Dance1")
            # if dance==[]:
            #     animatedSpeechProxy.say("Teach me to dance first!", configuration)
            # else:
            tmp = threading.Thread(target=playsound)
            tmp.start()
            doMouvement(dance)
        else:       # end
            animatedSpeechProxy.say("bye !", configuration)
            break
    motion_service.rest()






