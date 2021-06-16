import mediapipe as mp
import cv2
import time

mpDraw=mp.solutions.drawing_utils
mpPose=mp.solutions.pose
pose=mpPose.Pose()


cap=cv2.VideoCapture('posing5.mp4')


pTime=0



frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)

result = cv2.VideoWriter('posed5.avi',cv2.VideoWriter_fourcc(*'MJPG'),10, size)

while(1):
    success,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id,lm in enumerate(results.pose_landmarks.landmark):
            h,w,c=img.shape
            print(id,lm)
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,'FPS:'+str(int(fps)+40),(70,110),cv2.FONT_HERSHEY_DUPLEX,2,(255,0,255),3)
    result.write(img)
    img1 = cv2.resize(img, (800,800))

    cv2.imshow('IMAGE',img1)
    cv2.waitKey(1)


cap.release()
result.release()