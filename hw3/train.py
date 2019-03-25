import numpy as np
import sys
# For NN
from keras.models import Sequential
# For fully connection, dropout, and activation
from keras.layers.core import Dense,Dropout,Activation
# For convolution, pooling, and connection before fully connect
from keras.layers import Conv2D, MaxPooling2D,Flatten,AveragePooling2D
# For using optimizers
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

# input argument
train_file = sys.argv[1]


# read data
# please note that the first element in each line have label
# The reason why I use str type is that the first element of each row is str type
data_in = np.genfromtxt(fname=train_file,skip_header=1,dtype=str,delimiter=' ')
# number of train data
num_train = len(data_in)

# initialize label
label = np.zeros(num_train)
#print(num_train)
for i in range(num_train) :
  label[i] = data_in[i,0].split(',')[0]
  data_in[i,0] = data_in[i,0].split(',')[1]

# This step is copy reference, so memory won't double
feature = data_in

# For one-hot
label = np_utils.to_categorical(label,7)


# reshaping for convolution
feature = np.reshape(feature,(num_train,48,48,1))

# np.save('train_X.npy', feature)
# sys.exit(0)

feature = feature.astype(float)
feature = feature/255


# split data
valid_feature = feature[:2000]
valid_label = label[:2000]
train_feature = feature[2000:]
train_label = label[2000:]


# build model
print("Start to build model")
model = Sequential()

model.add(Conv2D(64,(5,5),input_shape = (48,48,1), activation = 'relu'))
model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25)) 

model.add(Conv2D(128,(3,3), activation = 'relu'))
model.add(Conv2D(128,(3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(256,(3,3), activation = 'relu'))
model.add(Conv2D(256,(3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.35))

model.add(Flatten())


model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5)) 


  
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5)) 



model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5)) 

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(units=7,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen.fit(train_feature)
model.fit_generator(datagen.flow(train_feature,train_label,batch_size=128),steps_per_epoch=len(feature)/4,epochs=50)
# model.fit(feature,label,batch_size=10,epochs=50)
score = model.evaluate(valid_feature,valid_label)
print('Total loss on testing set : ',score[0])
print('accuracy of testing set : ',score[1])
print('done')


# read testing data
test_file = sys.argv[2] 
test_in = np.genfromtxt(fname=test_file,skip_header=1,dtype=str,delimiter=' ')
num_test = len(test_in)

for i in range(num_test) :
  test_in[i,0] = test_in[i,0].split(',')[1]


# reshaping testing data
test_in = np.reshape(test_in,(num_test,48,48,1))
test_in = test_in.astype(float)
test_in = test_in/255


result = model.predict(test_in)

f = open('default.csv','w')
f.write("id,label\n")
for i in range(len(result)) :
  f.write(str(i) + ',' + str(int(np.argmax(result[i]))) + '\n')

f.close()
print('end')

