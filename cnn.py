import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import pandas as pd

image_dir="Dataset_Celebrities/cropped"
lionel_messi_images=os.listdir(image_dir+ '/lionel_messi')
maria_sharapova_images=os.listdir(image_dir+ '/maria_sharapova')
roger_federer_images=os.listdir(image_dir+ '/roger_federer')
serena_williams_images=os.listdir(image_dir+ '/serena_williams')
virat_kohli_images=os.listdir(image_dir+ '/virat_kohli')

print("--------------------------------------\n")
 
'''print('The length of NO Tumor images is',len(no_tumor_images))
print('The length of Tumor images is',len(yes_tumor_images))
print("--------------------------------------\n")'''


dataset=[]
label=[]
img_siz=(128,128)


def whole_dataset(index,folder,path,dataset,label,image_dir,img_siz):
    for j,image_name in tqdm(enumerate(folder)):
        if(image_name.split('.')[1]=='png'):
            image=cv2.imread(image_dir+ path+ image_name)
            image=Image.fromarray(image,'RGB')
            image=image.resize(img_siz)
            dataset.append(np.array(image))
            label.append(index)

    return dataset,label

list_folders=[lionel_messi_images,maria_sharapova_images,roger_federer_images,serena_williams_images,virat_kohli_images]
list_directories=['/lionel_messi/','/maria_sharapova/','/roger_federer/','/serena_williams/','/virat_kohli/']
labels=[0,1,2,3,4]

for index,folder,path in zip(labels,list_folders,list_directories):
    whole_dataset(index,folder,path,dataset,label,image_dir,img_siz)


        
dataset=np.array(dataset)
label = np.array(label)

print("--------------------------------------\n")
print('Dataset Length: ',len(dataset))
print('Label Length: ',len(label))
print("--------------------------------------\n")


print("--------------------------------------\n")
print("Train-Test Split")
x_train,x_test,y_train,y_test=train_test_split(dataset,label,test_size=0.2,random_state=42)
print("--------------------------------------\n")

print("--------------------------------------\n")
print("Normalaising the Dataset. \n")



x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)

print("--------------------------------------\n")


model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu',input_shape=(128,128,3)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),  # Additional dense layer
  tf.keras.layers.Dense(5, activation='softmax')
])



model.summary()
print("--------------------------------------\n")

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])


print("--------------------------------------\n")
early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', mode='min', patience=10)
print("Training Started.\n")
history=model.fit(x_train,y_train,epochs=30,batch_size =128,validation_split=0.1)
print("Training Finished.\n")
print("--------------------------------------\n")



print("--------------------------------------\n")
print("Model Evalutaion Phase.\n")
loss,accuracy=model.evaluate(x_test,y_test)
print(f'Accuracy: {round(accuracy*100,2)}')
print("--------------------------------------\n")

print("Model Evalutaion Phase.\n")
loss,accuracy=model.evaluate(x_test,y_test)
print(f'Accuracy: {round(accuracy*100,2)}')
print("--------------------------------------\n")
y_pred=model.predict(x_test)
y_pred = (y_pred > 0.5).astype(int)

print("--------------------------------------\n")
print("Model Prediction.\n")

def make_prediction(img,model):
    img=cv2.imread(img)
    img=Image.fromarray(img)
    img=img.resize((128,128))
    img=np.array(img)
    input_img = np.expand_dims(img, axis=0)
    prediction = model.predict(input_img)
    predicted_class = np.argmax(prediction,axis=1)[0]
    class_name = ['lionel_messi','maria_sharapova','roger_federer','serena_williams','virat_kohli']
    predicted_class_name = class_name[predicted_class]
    return predicted_class_name

print(make_prediction('Dataset_Celebrities/cropped/virat_kohli/virat_kohli10.png',model))
print('--------------------------------------------------------')





