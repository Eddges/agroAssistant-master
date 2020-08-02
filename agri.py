#!/usr/bin/env python
# coding: utf-8

# In[ ]:




# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm
from tqdm import tqdm_notebook


# In[ ]:


pepper_train = pd.read_csv('data.csv')
potato_train = pd.read_csv('potato_data.csv')
tomato_train = pd.read_csv('tomato_data.csv')


# In[ ]:


pepper_train.head()


# In[ ]:


pepper_train_image = []
potato_train_image = []
tomato_train_image = []

for i in tqdm_notebook(range(1,pepper_train.shape[0]+1)):
    #img = image.load_img('train/'+train['id'][i].astype('str')+'.png', target_size=(28,28,1), grayscale=True)
    img = image.load_img("C:\\Users\\Shekhar\\Desktop\\proj\\pepper\\("+str(i)+")"+".jpg" , target_size=(28,28,1), color_mode = "grayscale")
    #img = image.load_img("C:\\Users\\Nishit\\Desktop\\proj\\pepper\\ ("+str(i)+")" , target_size=(28,28,1), color_mode = "grayscale")
    img = image.img_to_array(img)
    img = img/255
    pepper_train_image.append(img)
X1 = np.array(pepper_train_image)

for i in tqdm_notebook(range(1,potato_train.shape[0]+1)):
    img = image.load_img("C:\\Users\\Shekhar\\Desktop\\proj\\potato\\("+str(i)+")"+".jpg" , target_size=(28,28,1), color_mode = "grayscale")
    img = image.img_to_array(img)
    img = img/255
    potato_train_image.append(img)
X2 = np.array(potato_train_image)

for i in tqdm_notebook(range(1,tomato_train.shape[0]+1)):
    img = image.load_img("C:\\Users\\Shekhar\\Desktop\\proj\\tomato\\("+str(i)+")"+".jpg" , target_size=(28,28,1), color_mode = "grayscale")
    img = image.img_to_array(img)
    img = img/255
    tomato_train_image.append(img)
X3 = np.array(tomato_train_image)


# In[ ]:


y1 = pepper_train.iloc[:,1:2].values
y1 = to_categorical(y1)

y2 = potato_train.iloc[:,1:2].values
y2 = to_categorical(y2)

y3 = tomato_train.iloc[:,1:2].values
y3 = to_categorical(y3)


# In[ ]:


X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, random_state=42, test_size=0.2)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, random_state=42, test_size=0.2)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, random_state=42, test_size=0.2)


# In[ ]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))


# In[ ]:


model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])


# In[ ]:


for i in tqdm_notebook(range(50)):
    model.fit(X1, y1, epochs=1,verbose=False, validation_data=(X1_test, y1_test))
for i in tqdm_notebook(range(20)):
    model.fit(X2, y2, epochs=1,verbose=False, validation_data=(X2_test, y2_test))
for i in tqdm_notebook(range(5)):
    model.fit(X3, y3, epochs=1,verbose=False, validation_data=(X3_test, y3_test))


# In[ ]:


pepper_test = np.array(X1_test)
potato_test = np.array(X2_test)
tomato_test = np.array(X3_test)


# In[ ]:


pepper_prediction = model.predict_classes(X1_test)
potato_prediction = model.predict_classes(X2_test)
tomato_prediction = model.predict_classes(X3_test)


# In[ ]:


pepper_scores = model.evaluate(X1_test, y1_test,verbose=1)#, batch_size=batch_size)
potato_scores = model.evaluate(X2_test, y2_test,verbose=1)
tomato_scores = model.evaluate(X3_test, y3_test,verbose=1)


# In[ ]:


tomato_scores


# In[ ]:


potato_scores


# In[ ]:


pepper_scores


# In[ ]:





# In[ ]:





# In[ ]:




