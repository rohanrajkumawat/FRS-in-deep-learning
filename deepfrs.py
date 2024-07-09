import cv2
import os
# Disable oneDNN optimizations for TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from tqdm import tqdm
from random import shuffle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Function to generate a dataset
def generate_dataset():
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y+h, x:x+w]
        return cropped_face

    cap = cv2.VideoCapture(0)
    img_id = 0

    while True:
        ret, frame = cap.read()
        if face_cropped(frame) is not None:
            img_id += 1
            face = cv2.resize(face_cropped(frame), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = "data/Rohan." + str(img_id) + ".jpg"
            file_name_path = "Images for visualization/" + str(img_id) + ".jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Cropped Face", face)
            if cv2.waitKey(1) == 13 or int(img_id) == 21:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Collecting samples is completed !!!")

# generate_dataset()
# create a data folder

# Function to create a label
def my_label(image_name):
    name = image_name.split('.')[-2]
    if name == "kana":
        return 0
    elif name == "Rohan":
        return 1

# Function to create data
def my_data():
    data = []
    for img in tqdm(os.listdir("data")):
        path = os.path.join("data", img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (50, 50))
        data.append([np.array(img_data), my_label(img)])
    shuffle(data)
    return data

data = my_data()
train = data[:16]
test = data[16:]

X_train = np.array([i[0] for i in train]).reshape(-1, 50, 50, 1)
y_train = np.array([i[1] for i in train])
y_train = to_categorical(y_train)

X_test = np.array([i[0] for i in test]).reshape(-1, 50, 50, 1)
y_test = np.array([i[1] for i in test])
y_test = to_categorical(y_test)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Create the model
model = Sequential([
    Conv2D(32, (5, 5), activation='relu', input_shape=(50, 50, 1)),
    MaxPooling2D(pool_size=(5, 5)),
    Conv2D(64, (5, 5), activation='relu'),
    MaxPooling2D(pool_size=(5, 5)),
    Conv2D(128, (5, 5), activation='relu'),
    MaxPooling2D(pool_size=(5, 5)),
    Conv2D(64, (5, 5), activation='relu'),
    MaxPooling2D(pool_size=(5, 5)),
    Conv2D(32, (5, 5), activation='relu'),
    MaxPooling2D(pool_size=(5, 5)),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.8),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=12, validation_data=(X_test, y_test))

# Function to visualize the data and make a prediction
def data_for_visualization():
    Vdata = []
    for img in tqdm(os.listdir("Images for visualization")):
        path = os.path.join("Images for visualization", img)
        img_num = img.split('.')[0]
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (50, 50))
        Vdata.append([np.array(img_data), img_num])
    shuffle(Vdata)
    return Vdata

Vdata = data_for_visualization()

fig = plt.figure(figsize=(20, 20))
for num, data in enumerate(Vdata[:20]):
    img_data = data[0]
    y = fig.add_subplot(5, 5, num + 1)
    image = img_data
    data = img_data.reshape(1, 50, 50, 1)
    model_out = model.predict(data)[0]
    
    if np.argmax(model_out) == 0:
        my_label = 'kana'
    elif np.argmax(model_out) == 1:
        my_label = 'Rohan'

    y.imshow(image, cmap='gray')
    plt.title(my_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()
