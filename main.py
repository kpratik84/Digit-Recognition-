% pylab inline
import os
import random

import pandas as pd
from scipy.misc import imread

root_dir = os.path.abspath('.')
data_dir = '/Users/kpratik84/Documents/Deep Learning/mnist/Train/'

train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

i = random.choice(train.index)

img_name = train.filename[i]
img = imread(os.path.join(data_dir, 'Images/train', img_name))

imshow(img)

temp = []
for img_name in train.filename:
    img_path = os.path.join(data_dir, 'Images/train', img_name)
    img = imread(img_path)
    # img = imresize(img, (32, 32))
    img = img.astype('float32') # this will help us in later stage
    temp.append(img)

train_x = np.stack(temp)

temp = []
for img_name in test.filename:
    img_path = os.path.join(data_dir, 'Images/test', img_name)
    img = imread(img_path)
    # img = imresize(img, (32, 32))
    img = img.astype('float32') # this will help us in later stage
    temp.append(img)

test_x = np.stack(temp)

train_x = train_x / 255.
test_x = test_x / 255.


train.label.value_counts(normalize=True)

import keras
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
train_y = lb.fit_transform(train.label)
train_y = keras.utils.np_utils.to_categorical(train_y)



from keras.models import Sequential
from keras.layers import Dense, Flatten, InputLayer

input_num_units = (28, 28, 4)
hidden_num_units = 500
output_num_units = 10
epochs = 5
batch_size = 128

model = Sequential()
model.add(InputLayer(input_shape=input_num_units))
model.add(Flatten())
model.add(Dense(units=hidden_num_units, activation='relu'))
model.add(Dense(units=output_num_units, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
# Fit the model
model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,  verbose=1)
# calculate predictions
predictions = model.predict(test_x)


pred = model.predict_classes(test_x)
pred = lb.inverse_transform(pred)

i = random.choice(test.index)
img_name = test.filename[i]
img = imread(os.path.join(data_dir, 'Images/test', img_name))

imshow(img)
print (pred[i])





