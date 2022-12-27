import json
import pandas as pd
import cv2
from keras.models import load_model
import numpy as np
from datetime import datetime as dt
import pytz
from utils.path_helper import haarCascade_file, trained_model, labels, attendance_database

# loading the haar cascade classifier to detect the front face
facedetect = cv2.CascadeClassifier(haarCascade_file)
# Starting the camera
# Setting up the parameters
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font = cv2.FONT_HERSHEY_COMPLEX
# Match Percentage Threshold value
MATCH_THRESHOLD = 90

# Loading the trained model
model = load_model(trained_model / 'MobileNet_untuned_model.h5')

# Loading the labels
file = open(labels / 'labels.json', "r")
label = json.loads(file.read())
label_result = {v: k for k, v in label.items()}


# Function to create the attendance record
def record_attendance(predicted_name):
    current_time = dt.now(pytz.timezone('Asia/Kolkata'))
    record_det = {
        'Name': predicted_name,
        'Date': current_time.date(),
        'Current_Time': current_time.time()

    }
    return record_det


print(label_result)
attendance_record = []
time = dt.now(pytz.timezone('Asia/Kolkata'))
# Detecting Face and marking the attendance
while True:
    success, inputImage = cap.read()
    faces = facedetect.detectMultiScale(inputImage, 1.3, 5)
    for x, y, w, h in faces:
        crop_img = inputImage[y:y + h, x:x + h]
        img = cv2.resize(crop_img, (224, 224))
        img = img.reshape(1, 224, 224, 3)
        prediction = model.predict(img)
        classIndex = np.argmax(model.predict(img), axis=-1)
        matchedName = str(label_result[int(classIndex)]).split(' ')
        predictedName = ' '.join([word.title() for word in matchedName])
        matchScore = np.amax(prediction)
        matchPercentage = round(matchScore * 100, 2)
        cv2.rectangle(inputImage, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(inputImage, (x, y - 40), (x + w, y), (0, 255, 0), -2)
        cv2.putText(inputImage, predictedName, (x, y - 10), font, 0.75, (255, 255, 255), 1,
                    cv2.LINE_AA)

        cv2.putText(inputImage, str(matchPercentage) + "%", (180, 75), font, 0.75, (255, 0, 0), 2,
                    cv2.LINE_AA)
        print(matchPercentage)
        # checking the match percentage is greater or not than the threshold
        if matchPercentage > MATCH_THRESHOLD:
            attendance_record.append(record_attendance(predictedName))
        else:
            print('No Match Found in the Dataset')

    cv2.imshow("Result", inputImage)
    # Pres Esc to close
    key = cv2.waitKey(1)
    if key == 27:
        break

# Closing the camera input
cap.release()
cv2.destroyAllWindows()
# Saving the attendance Record
record_df = pd.DataFrame(attendance_record)
record_df.drop_duplicates(["Date"],
                          keep="first",
                          inplace=True)
record_df.to_csv(attendance_database / f'Attendance_Record_{time.date()}', index=False)
