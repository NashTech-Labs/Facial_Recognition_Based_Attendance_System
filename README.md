# Facial Recognition Based Attendance System

This repository content will help you with the following operations:

- Create facial image dataset.
- Train a classification model using above created facial data.
- Mark your attendance using facial recognition.

### To install the dependencies/requirements of the project

- `pip install -r requirements.txt`

## Create Facial Image Dataset

We have created a script which will automatically generate facial image data.

- [Create-Facial-Data](database/create_facial_data.py)
- Above script will capture 1500 image of your face. 
- You can increase or decrease the count of the image data by tweaking the parameters.

## Train/Fine Tune the image classification model.

We are using `MobileNetV2` a powerful pre-trained image classification model.
We will be retraining this pre-trained model by excluding its top layer/prediction layer, using as a feature extractor.
Adding our own classification layer to it according to our dataset (that means number of faces to be classified).

- [Model-Training](model/train_classification_model.py)
- This script will help you train your model and saved the trained model.
- You have to specify the number of outputs in the dense layer `tf.keras.layers.Dense(<numebr of faces/outputs>, activation='softmax')`
- Then you can use the `main function` to invoke the training of the model.

## Test the Model (Mark Attendance)

After training the model we will load the trained model and test it.

- [Mark-Attendance](facial_recoginition/mark_attendance.py)
- When you run the script, it will capture you face and, predict the name of the input face and the matching percentage.
- We set a threshold value, if the `match percentage` is greater than the threshold.
- Then it will create an entry to the attendance record and store the csv in [Attendance Database](database/attendance_database) directory.
- Press `ESC` to close the window.


# Thank You !!