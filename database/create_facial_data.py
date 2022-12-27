import cv2
import os
from utils.path_helper import haarCascade_file

# Starting the camera
video = cv2.VideoCapture(0)

# loading the haar cascade classifier to detect the frontal face
facedetect = cv2.CascadeClassifier(haarCascade_file)

count = 0

nameID = str(input("Please Enter Your Name: ")).lower()

path = 'images/' + nameID

isExist = os.path.exists(path)
# checking if the directory exist. if it does not exist it will create images directory
if isExist:
    print("Entered Name Already Exists")
    nameID = str(input("Please Enter Your Name Again: "))
else:
    os.makedirs(path)

while True:
    ret, frame = video.read()
    faces = facedetect.detectMultiScale(frame, 1.3, 5)
    for x, y, w, h in faces:
        count = count + 1
        name = './images/' + nameID + '/' + str(count) + '.jpg'
        print("Creating Image Dataset-----------------------" + name)
        cv2.imwrite(name, frame[y:y + h, x:x + w])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imshow("Image Frame", frame)
    cv2.waitKey(1)
    if count > 1499:
        break
video.release()
cv2.destroyAllWindows()
