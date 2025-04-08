#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split
from PIL import Image
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall, AUC
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import math


# In[2]:


import cv2
cap = cv2.VideoCapture('C:\\Users\\acer\\OneDrive\\Documents\\College\\Semesters\\Sem10\\Project\\trisgym.mp4')
frameRate = cap.get(5)  
count = 157
save_dir = 'C:\\Users\\acer\\OneDrive\\Documents\\College\\Semesters\\Sem10\\Project\\TristanFrame'

while(cap.isOpened()):
    frameId = cap.get(1)  
    ret, frame = cap.read()
    if not ret:
        break
    if frameId % math.floor(frameRate) == 0:
        filename = save_dir + f"frame{count}.jpg"
        count += 1
        cv2.imwrite(filename, frame)

cap.release()


# In[3]:


import cv2
import numpy as np
import os
import glob

image_dir = r'C:\Users\acer\OneDrive\Documents\College\Semesters\Sem10\Project\faces\TristanFrame'

if not os.path.exists(image_dir):
    print(f"The directory {image_dir} does not exist.")
else:
    print(f"The directory {image_dir} was found.")

target_size = (224, 224)

access_granted = []

image_files = glob.glob(os.path.join(image_dir, '*.jpg'))

if not image_files:
    print(f"No .jpg files found in the directory {image_dir}.")
else:
    print(f"Found {len(image_files)} .jpg files in the directory {image_dir}.")

for img_path in image_files:
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Failed to read the image {img_path}.")
        continue

    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    
    access_granted.append(img)

access_granted = np.array(access_granted)

if access_granted.size > 0:
    print(f"Loaded {access_granted.shape[0]} images.")
else:
    print("No images were loaded into the array.")


# In[4]:


import cv2
import numpy as np
import os
import glob

image_dir = r'C:\Users\acer\OneDrive\Documents\College\Semesters\Sem10\Project\lfw_funneled'

# Make sure the directory exists
if not os.path.exists(image_dir):
    print(f"The directory {image_dir} does not exist.")
else:
    print(f"The directory {image_dir} was found.")

target_size = (224, 224)
access_denied = []

image_files = []
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('.jpg'):
            image_files.append(os.path.join(root, file))

if not image_files:
    print(f"No .jpg files found in the directory {image_dir} and its subdirectories.")
else:
    print(f"Found {len(image_files)} .jpg files in the directory {image_dir} and its subdirectories.")

for img_path in image_files:
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read the image {img_path}.")
        continue

    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    access_denied.append(img)
    if len(access_denied) >= 164:
            break

access_denied= np.array(access_denied)

if access_denied.size > 0:
    print(f"Loaded {access_denied.shape[0]} images.")
else:
    print("No images were loaded into the array.")


# In[5]:


import cv2
import numpy as np
import os
import glob

image_dir = r'C:\Users\acer\OneDrive\Documents\College\Semesters\Sem10\Project\TristanValidation'

if not os.path.exists(image_dir):
    print(f"The directory {image_dir} does not exist.")
else:
    print(f"The directory {image_dir} was found.")

target_size = (224, 224)

valid_pre = []

image_files = glob.glob(os.path.join(image_dir, '*.jpg'))

if not image_files:
    print(f"No .jpg files found in the directory {image_dir}.")
else:
    print(f"Found {len(image_files)} .jpg files in the directory {image_dir}.")

for img_path in image_files:
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Failed to read the image {img_path}.")
        continue

    img = cv2.resize(img, target_size)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = img.astype('float32') / 255.0
    
    valid_pre.append(img)

valid_pre = np.array(valid_pre)

if valid_pre.size > 0:
    print(f"Loaded {valid_pre.shape[0]} images.")
else:
    print("No images were loaded into the array.")


# In[6]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

augmentations_per_image = 20  

tris_images = []
tris_labels = []

for img in access_granted:  
    img_tensor = np.expand_dims(img, axis=0)  
    count = 0  
    
    
    for aug_img in datagen.flow(img_tensor, batch_size=1):
        aug_img = aug_img[0]  
        tris_images.append(aug_img)  
        tris_labels.append(1)  
        count += 1
        if count >= augmentations_per_image: 
            break

tris_images = np.array(tris_images)
tris_labels = np.array(tris_labels)


# In[7]:


from keras.preprocessing.image import ImageDataGenerator
import numpy as np

access_denied_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

augmented_access_denied_images = []
augmented_access_denied_labels = []  
for _ in range(2): 
    for img in access_denied:
        img = img.reshape((1, ) + img.shape)  
        for batch in access_denied_datagen.flow(img, batch_size=1):
            augmented_access_denied_images.append(batch[0])
            augmented_access_denied_labels.append(0)
            break 

augmented_access_denied_images = np.array(augmented_access_denied_images)
augmented_access_denied_labels = np.array(augmented_access_denied_labels)



# In[8]:


access_granted_size = len(tris_images)
access_denied_size = len(access_denied)

if access_granted_size < access_denied_size:
    replicate_indices = np.random.choice(tris_images.shape[0], access_denied_size - access_granted_size, replace=True)
    tris_images_over = np.concatenate((tris_images, tris_images[replicate_indices]), axis=0)
    access_denied_over = access_denied
else:
    replicate_indices = np.random.choice(access_denied.shape[0], access_granted_size - access_denied_size, replace=True)
    access_denied_over = np.concatenate((access_denied, access_denied[replicate_indices]), axis=0)
    tris_images_over = tris_images

all_images_balanced = np.concatenate((tris_images_over, access_denied_over), axis=0)
all_labels_balanced = np.concatenate((np.ones(len(tris_images_over)), np.zeros(len(access_denied_over))), axis=0)

indices = np.arange(all_images_balanced.shape[0])
np.random.shuffle(indices)
all_images_balanced = all_images_balanced[indices]
all_labels_balanced = all_labels_balanced[indices]
print("Access denied size")
print(access_denied_size)
print("Access granted size")
print(access_granted_size)


# In[9]:


validation_image_dir = r'C:\Users\acer\OneDrive\Documents\College\Semesters\Sem10\Project\TristanValidation'
validation_target_size = (224, 224)
validation_images = []

for img_path in glob.glob(os.path.join(validation_image_dir, '*.jpg')):
    img = cv2.imread(img_path)
    img = cv2.resize(img, validation_target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    validation_images.append(img)

validation_images = np.array(validation_images)
validation_labels = np.ones(len(validation_images))  


# In[10]:


from sklearn.model_selection import train_test_split #ADAM OPTIMIZER 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight
import numpy as np

all_images = np.concatenate((access_granted, access_denied))
all_labels = np.concatenate((np.ones(len(access_granted)), np.zeros(len(access_denied))))

classes = np.unique(all_labels)

weights = compute_class_weight(class_weight='balanced', classes=classes, y=all_labels)
class_weights = {classes[i]: weights[i] for i in range(len(classes))}

X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train) # 0.25 x 0.8 = 0.2

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

history = model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val), class_weight=class_weights, callbacks=[early_stopping, model_checkpoint])

model.evaluate(X_test, y_test)

model.save('final_model.h5')


# In[11]:


from tensorflow.keras.optimizers import SGD #MODEL WITH SGD 

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

sgd_optimizer = SGD(learning_rate=0.01, momentum=0.9)

model.compile(optimizer=sgd_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

history = model.fit(
    X_train, y_train,
    epochs=30,
    validation_data=(X_val, y_val),
    class_weight=class_weights,
    callbacks=[early_stopping, model_checkpoint]
)

model.evaluate(X_test, y_test)

model.save('final_model_with_SGD.h5')


# In[12]:


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(all_images, all_labels, test_size=0.2, stratify=all_labels, random_state=42)


# In[ ]:





# In[13]:


from tensorflow.keras.layers import BatchNormalization, Dropout, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
import tensorflow as tf

def build_model(input_shape=(224, 224, 3)):
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.5),  
        Dense(1, activation='sigmoid')
    ])
    return model

def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

augmented_data_generator = datagen.flow(tris_images, tris_labels, batch_size=32)
validation_labels = np.ones(len(valid_pre)) 


# In[14]:


from tensorflow.keras.optimizers import SGD

model = build_model()
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    augmented_data_generator,
    steps_per_epoch=len(access_granted) // 32,
    validation_data=(validation_images, validation_labels),
    epochs=10,
    callbacks=[early_stopping, lr_callback]
)


# In[15]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.show()


# In[16]:


model.save('C:\\Users\\acer\\OneDrive\\Documents\\College\\Semesters\\Sem10\\Project\\save\\model.h5')
#SGD 1 


# In[ ]:


import cv2 #OPTIMIZATION WITH SGD 
import numpy as np
from keras.models import load_model

model = load_model('C:\\Users\\acer\\OneDrive\\Documents\\College\\Semesters\\Sem10\\Project\\save\\model.h5')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(face, target_size=(224, 224)):
    face = cv2.resize(face, target_size)
    face = face.astype('float32')
    face = np.expand_dims(face, axis=0)
    face /= 255  
    return face

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        preprocessed_face = preprocess_face(face)

        prediction = model.predict(preprocessed_face)

        if prediction > 0.2:
            print("Access Granted")
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            print("Access Denied")
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:


import cv2 #OPTIMIZATION WITH SGD 
import numpy as np
from keras.models import load_model

model_path = 'C:\\Users\\acer\\OneDrive\\Documents\\College\\Semesters\\Sem10\\Project\\faces\\final_model_with_SGD.h5'
model = load_model(model_path)

face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

def preprocess_face(face, target_size=(224, 224)):
    face = cv2.resize(face, target_size)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    return face

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        preprocessed_face = preprocess_face(face)

        prediction = model.predict(preprocessed_face)

        probability = prediction[0][0]
        if probability > 0.2:  
            access_label = "Access Granted"
            rectangle_color = (0, 255, 0)
        else:
            access_label = "Access Denied"
            rectangle_color = (0, 0, 255)
        
        cv2.putText(frame, f"{access_label}: {probability:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, rectangle_color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), rectangle_color, 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

true_labels = np.random.choice([0, 1], size=100)  

predictions = np.random.choice([0, 1], size=100)  

cm = confusion_matrix(true_labels, predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Denied', 'Granted'], yticklabels=['Denied', 'Granted'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# 

# 

# In[ ]:


import matplotlib.pyplot as plt

# Accuracy plot
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Loss plot
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

def plot_model_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Train Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Train Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.show()

plot_model_history(history)


# In[ ]:


import cv2
import numpy as np #THIS ONE WORKS PERFECTLY DO NOT DELETE AT ALL ADAM OPTIMIZER 
from keras.models import load_model

model_path = 'C:\\Users\\acer\\OneDrive\\Documents\\College\\Semesters\\Sem10\\Project\\faces\\final_model.h5'
model = load_model(model_path)

face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

def preprocess_face(face, target_size=(224, 224)):
    face = cv2.resize(face, target_size)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    return face

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        preprocessed_face = preprocess_face(face)

        prediction = model.predict(preprocessed_face)

        probability = prediction[0][0]
        if probability > 0.9:  
            access_label = "Access Granted"
            rectangle_color = (0, 255, 0)
        else:
            access_label = "Access Denied"
            rectangle_color = (0, 0, 255)
        
        cv2.putText(frame, f"{access_label}: {probability:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, rectangle_color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), rectangle_color, 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:


import cv2
import numpy as np #WITH SGD OPTIMIZER 
from keras.models import load_model

model_path = 'C:\\Users\\acer\\OneDrive\\Documents\\College\\Semesters\\Sem10\\Project\\faces\\final_model_with_SGD.h5'
model = load_model(model_path)

face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

def preprocess_face(face, target_size=(224, 224)):
    face = cv2.resize(face, target_size)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    return face

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        preprocessed_face = preprocess_face(face)

        prediction = model.predict(preprocessed_face)

        probability = prediction[0][0]
        if probability > 0.9:  
            access_label = "Access Granted"
            rectangle_color = (0, 255, 0)
        else:
            access_label = "Access Denied"
            rectangle_color = (0, 0, 255)
        
        cv2.putText(frame, f"{access_label}: {probability:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, rectangle_color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), rectangle_color, 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




