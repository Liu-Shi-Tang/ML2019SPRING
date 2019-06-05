import numpy as np
import sys
import keras
# For NN
from keras.models import Sequential,load_model
# For using optimizers
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
# For convolution, pooling, and connection before fully connect
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D,  BatchNormalization, LeakyReLU, DepthwiseConv2D, Flatten, AveragePooling2D, Activation
# Callbacks
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger

# parsing input data #################################################################
def parsingTrainingData(file_name) :
  # read data
  # please note that the first element in each line have label
  # The reason why I use str type is that the first element of each row is str type
  data_in = np.genfromtxt(fname=file_name,skip_header=1,dtype=str,delimiter=' ')
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
  feature = feature.astype(float)
  feature = feature/255
  
  valid_feature = []
  valid_label = []
  train_feature = []
  train_label = []
  for i in range(len(feature)) :
    if i % 15 == 0 :
      valid_feature.append(feature[i])
      valid_label.append(label[i]) 
    else :
      train_feature.append(feature[i])
      train_label.append(label[i])

  valid_feature = np.array(valid_feature)
  valid_label = np.array(valid_label)
  train_feature = np.array(train_feature)
  train_label = np.array(train_label)


  # split data
#  valid_feature = feature[:2000]
#  valid_label = label[:2000]
#  train_feature = feature[2000:]
#  train_label = label[2000:]
  return train_feature,train_label,valid_feature,valid_label


# read training data #############################################################################
train_file = sys.argv[1]
train_feature,train_label,valid_feature,valid_label = parsingTrainingData(train_file)

# build model #####################################################################################
def getModel() :
  
  model = Sequential()
  
  model.add(Conv2D(32,(5,5),input_shape = (48,48,1), activation = 'linear', padding='same',kernel_initializer='glorot_normal'))
  model.add(BatchNormalization())
  model.add(LeakyReLU(alpha=0.2))
  model.add(AveragePooling2D(pool_size=(2,2)))
  model.add(Dropout(0.02)) 
    
  model.add(DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='linear'))
  model.add(BatchNormalization())
  model.add(LeakyReLU(alpha=0.2))
  model.add(Conv2D(64,(1,1), activation = 'linear', padding='same',kernel_initializer='glorot_normal'))
  model.add(BatchNormalization())
  model.add(LeakyReLU(alpha=0.2))
  model.add(AveragePooling2D(pool_size=(2,2))) 
  model.add(Dropout(0.03)) 
  
  model.add(DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='linear'))
  model.add(BatchNormalization())
  model.add(LeakyReLU(alpha=0.2))
  model.add(Conv2D(168,(1,1), activation = 'linear', padding='same',kernel_initializer='glorot_normal'))
  model.add(BatchNormalization())
  model.add(LeakyReLU(alpha=0.2))
  model.add(AveragePooling2D(pool_size=(2,2))) 
  model.add(Dropout(0.04)) 
  
  model.add(DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='linear'))
  model.add(BatchNormalization())
  model.add(LeakyReLU(alpha=0.2))
  model.add(Conv2D(56,(1,1), activation = 'linear', padding='same',kernel_initializer='glorot_normal'))
  model.add(BatchNormalization())
  model.add(LeakyReLU(alpha=0.2))
  model.add(AveragePooling2D(pool_size=(2,2))) 
  model.add(Dropout(0.02)) 
  
  
  model.add(Flatten())
  
  model.add(Dense(20,kernel_initializer='glorot_normal'))
  # model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  
  model.add(Dense(units=7,activation='softmax'))
  myOpt = Adam(lr=0.02)
  model.compile(loss='categorical_crossentropy',optimizer=myOpt,metrics=['accuracy'])
  
  
  return model

model = getModel()
model.summary()

csv_logger = CSVLogger('log.csv', append=False)
learning_rate = ReduceLROnPlateau(monitor='acc',factor = 0.2, patience=6, verbose=1, mode='auto', min_delta=1e-4,cooldown=0, min_lr=1e-8)
checkpoint = ModelCheckpoint(filepath='best.h5', monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=True,mode='auto',period=1)
early_stop = EarlyStopping(monitor='acc', patience=13, verbose=1, mode='auto',min_delta=0.00005 )

# For augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=[0.85,1.15],
    shear_range=0.2,
    horizontal_flip=True)

# datagen.fit(train_feature)
model.fit_generator(datagen.flow(train_feature,train_label,batch_size=128),
    steps_per_epoch=len(train_feature)/16,
    epochs=1000,
    validation_data=(valid_feature,valid_label),
    callbacks=[csv_logger,learning_rate,checkpoint,early_stop])

