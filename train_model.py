import kagglehub
import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Download dataset from Kaggle
dataset_path = kagglehub.dataset_download("omkargurav/face-mask-dataset")

print("Dataset downloaded to:", dataset_path)

# Dataset folders
with_mask_path = os.path.join(dataset_path, "data", "with_mask")
without_mask_path = os.path.join(dataset_path, "data", "without_mask")

categories = [with_mask_path, without_mask_path]

data = []
labels = []

# Read images
for label, category in enumerate(categories):
    
    for img in os.listdir(category):
        
        img_path = os.path.join(category, img)
        
        try:
            image = cv2.imread(img_path)
            image = cv2.resize(image, (64,64))   # smaller size for weak laptop
            
            data.append(image)
            labels.append(label)
            
        except:
            pass

data = np.array(data) / 255.0
labels = to_categorical(labels, 2)

print("Total images loaded:", len(data))

# Build CNN Model
model = Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(64,64,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(2,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train model
model.fit(data,labels,epochs=5,batch_size=32)

# Save trained model
model.save("mask_model.h5")

print("Model trained and saved as mask_model.h5")