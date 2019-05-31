import numpy as np
import sys
import keras
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

# for loading
import keras.models
from keras.models import load_model


# For establish model
import keras
# For NN
from keras.models import Sequential
# For fully connection, dropout, and activation
from keras.layers.core import Dense,Dropout,Activation
# For convolution, pooling, and connection before fully connect
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, GlobalMaxPooling2D, BatchNormalization, LeakyReLU, DepthwiseConv2D, Flatten, GlobalAveragePooling2D,AveragePooling2D
# For using optimizers
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger






# read testing data
test_file = sys.argv[1] 
test_in = np.genfromtxt(fname=test_file,skip_header=1,dtype=str,delimiter=' ')
num_test = len(test_in)

for i in range(num_test) :
  test_in[i,0] = test_in[i,0].split(',')[1]

# reshaping testing data
test_in = np.reshape(test_in,(num_test,48,48,1))
test_in = test_in.astype(float)
test_in = test_in/255


######################################################################################################################################

model = Sequential()

model.add(Conv2D(32,(5,5),input_shape = (48,48,1), activation = 'linear', padding='same',kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.2))
# model.add(DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='linear'))
# model.add(BatchNormalization())
# model.add(LeakyReLU(alpha=0.2))
# model.add(Conv2D(32,(1,1), activation = 'linear', padding='same',kernel_initializer='glorot_normal'))
# model.add(BatchNormalization())
# model.add(LeakyReLU(alpha=0.2))
# model.add(MaxPooling2D((2,2)))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Dropout(0.05)) 
  
model.add(DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='linear'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.2))
model.add(Conv2D(64,(1,1), activation = 'linear', padding='same',kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.2))
# model.add(DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='linear'))
# model.add(BatchNormalization())
# model.add(LeakyReLU(alpha=0.2))
# model.add(Conv2D(64,(1,1), activation = 'linear', padding='same',kernel_initializer='glorot_normal'))
# model.add(BatchNormalization())
# model.add(LeakyReLU(alpha=0.2))
# model.add(MaxPooling2D((2,2)))
model.add(AveragePooling2D(pool_size=(2,2))) 
model.add(Dropout(0.05)) 

model.add(DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='linear'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.2))
model.add(Conv2D(128,(1,1), activation = 'linear', padding='same',kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.2))
# model.add(DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='linear'))
# model.add(BatchNormalization())
# model.add(LeakyReLU(alpha=0.2))
# model.add(Conv2D(128,(1,1), activation = 'linear', padding='same',kernel_initializer='glorot_normal'))
# model.add(BatchNormalization())
# model.add(LeakyReLU(alpha=0.2))
# model.add(MaxPooling2D((2,2)))
model.add(AveragePooling2D(pool_size=(2,2))) 
model.add(Dropout(0.05)) 

model.add(DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='linear'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.2))
model.add(Conv2D(256,(1,1), activation = 'linear', padding='same',kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.2))
model.add(AveragePooling2D(pool_size=(2,2))) 
model.add(Dropout(0.05)) 





# model.add(DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='linear'))
# model.add(BatchNormalization())
# model.add(LeakyReLU(alpha=0.2))
# model.add(Conv2D(512,(1,1), activation = 'linear', padding='same',kernel_initializer='glorot_normal'))
# model.add(BatchNormalization())
# model.add(LeakyReLU(alpha=0.2))
# model.add(MaxPooling2D((2,2)))

# model.add(AveragePooling2D(pool_size=(2,2))) 
# model.add(Dropout(0.1)) 






#model.add(Dropout(0.2)) 




# model.add(Conv2D(128,(3,3), activation = 'relu', padding='same',kernel_initializer='glorot_normal'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D((2,2)))
# model.add(Dropout(0.3))
# 
# model.add(Conv2D(256,(3,3), activation = 'relu', padding='same',kernel_initializer='glorot_normal'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D((2,2)))
# model.add(Dropout(0.35))
# 
# model.add(Conv2D(512,(3,3), activation = 'relu', padding='same',kernel_initializer='glorot_normal'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D((2,2)))
# model.add(Dropout(0.4))


model.add(Flatten())

model.add(Dense(32,kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(Dense(64,kernel_initializer='glorot_normal'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dense(512,kernel_initializer='glorot_normal'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dropout(0.1)) 

# model.add(Dense(512,kernel_initializer='glorot_normal'))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5)) 



model.add(Dense(units=7,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=[0.8,1.2],
    shear_range=0.2,
    horizontal_flip=True)

model.summary()




######################################################################################################################################


model.load_weights('best.h5')

weights = model.get_weights()
# print("weights:")
# print(np.max(weights))
# print(np.min(weights))


# without std
result = model.predict(test_in)



# write result
result_file = sys.argv[2]


f = open(result_file,'w')
f.write("id,label\n")
for i in range(len(result)) :
  f.write(str(i) + ',' + str(int(np.argmax(result[i]))) + '\n')

f.close()
  
print('end')


