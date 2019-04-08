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
  
  # np.save('train_X.npy', feature)
  # sys.exit(0)
  
  feature = feature.astype(float)
  feature = feature/255
  
  # split data
  valid_feature = feature[:2000]
  valid_label = label[:2000]
  train_feature = feature[2000:]
  train_label = label[2000:]
 
  mean , std = np.mean(train_feature,axis=0) , np.std(train_feature,axis=0)

  train_feature = (train_feature - mean) / (std + 1e-20) 
  valid_feature = (valid_feature - mean) / (std + 1e-20)   
  np.save('nor_only_mean',mean)
  np.save('nor_only_std',std)

  return train_feature,train_label,valid_feature,valid_label,mean,std



# parsing testing data ###############################################################################
def parsingTestingData(file_name,mean,std) :
  test_in = np.genfromtxt(fname=file_name,skip_header=1,dtype=str,delimiter=' ')
  num_test = len(test_in)
  
  for i in range(num_test) :
    test_in[i,0] = test_in[i,0].split(',')[1]
  
  
  # reshaping testing data
  test_in = np.reshape(test_in,(num_test,48,48,1))
  test_in = test_in.astype(float)
  test_in = test_in/255

  test_in = (test_in - mean ) / (std + 1e-20)


  return test_in

# write result ###################################################################################
def writeResult(file_name,result) :
  f = open(file_name,'w')
  f.write("id,label\n")
  for i in range(len(result)) :
    f.write(str(i) + ',' + str(int(np.argmax(result[i]))) + '\n')
  
  f.close()
  


# read training data #############################################################################
train_file = sys.argv[1]
train_feature,train_label,valid_feature,valid_label,mean,std = parsingTrainingData(train_file)

# build model #####################################################################################

  
# First define my initializer
def myInit( shape , dtype=None) :
  return keras.initializers.RandomNormal(mean=0.0,stddev=0.05,seed=666)

  
print("Start to build model")
model = Sequential()

model.add(Conv2D(512,(5,5),input_shape = (48,48,1), activation = 'relu', padding='same',kernel_initializer=keras.initializers.RandomNormal(mean=0.0,stddev=0.05,seed=666)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25)) 

model.add(Conv2D(512,(5,5), activation = 'relu', padding='same',kernel_initializer=keras.initializers.RandomNormal(mean=0.0,stddev=0.05,seed=666)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(768,(3,3), activation = 'relu', padding='same',kernel_initializer=keras.initializers.RandomNormal(mean=0.0,stddev=0.05,seed=666)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.35))

model.add(Conv2D(512,(3,3), activation = 'relu', padding='same',kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.4))


model.add(Flatten())


model.add(Dense(512,kernel_initializer=keras.initializers.RandomNormal(mean=0.0,stddev=0.05,seed=666)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5)) 

model.add(Dense(512,kernel_initializer=keras.initializers.RandomNormal(mean=0.0,stddev=0.05,seed=666)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5)) 



model.add(Dense(units=7,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# datagen = ImageDataGenerator(
#     rotation_range=30,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     zoom_range=[0.8,1.2],
#     shear_range=0.2,
#     horizontal_flip=True)

model.summary()


mcp = keras.callbacks.ModelCheckpoint('mcp-nor-only-acc-{val_acc:.5f}.h5',
    monitor='val_acc',
    save_best_only=True,
    verbose=1,
    mode='auto',
    period=1)

es = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
    factor=0.1,
    patience=10,
    verbose=1,
    mode='auto',
    min_delta=0.000001,
    min_lr=0.0001)

# datagen.fit(train_feature)
history = model.fit(
    x=train_feature,
    y=train_label,
    batch_size=128,
    epochs=250,
    validation_data=(valid_feature,valid_label),
    callbacks=[mcp,es])
# model.fit(feature,label,batch_size=10,epochs=50)
score = model.evaluate(valid_feature,valid_label)
print('Total loss on testing set : ',score[0])
print('accuracy of testing set : ',score[1])
print('done')


# read testing data #######################################################################
test_file = sys.argv[2] 
test_in = parsingTestingData(test_file,mean,std)

# predict for testing data ################################################################
result = model.predict(test_in)


# write result ###########################################################################
writeResult('nor_only.csv',result)


# save model
model.save('nor_only_m.h5')
 
# save history of acc loss
np_val_acc = np.array(history.history['val_acc'])
np_tra_acc = np.array(history.history['acc'])
np_val_loss = np.array(history.history['val_loss'])
np_tra_loss = np.array(history.history['loss'])
np.save('nor_only_val_loss',np_val_loss)
np.save('nor_only_tra_loss',np_tra_loss)
np.save('nor_only_val_acc',np_val_acc)
np.save('nor_only_tra_acc',np_tra_acc)
print('end')

