import numpy as np
import sys
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
  return train_feature,train_label,valid_feature,valid_label



# parsing testing data ###############################################################################
def parsingTestingData(file_name) :
  test_in = np.genfromtxt(fname=file_name,skip_header=1,dtype=str,delimiter=' ')
  num_test = len(test_in)
  
  for i in range(num_test) :
    test_in[i,0] = test_in[i,0].split(',')[1]
  
  
  # reshaping testing data
  test_in = np.reshape(test_in,(num_test,48,48,1))
  test_in = test_in.astype(float)
  test_in = test_in/255

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
train_feature,train_label,valid_feature,valid_label = parsingTrainingData(train_file)

# build model #####################################################################################
print("Start to build model")
model = Sequential()

model.add(Conv2D(32,(3,3),input_shape = (48,48,1), activation = 'linear', padding='same',kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.2))
# model.add(DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='linear'))
# model.add(BatchNormalization())
# model.add(LeakyReLU(alpha=0.2))
# model.add(Conv2D(32,(1,1), activation = 'linear', padding='same',kernel_initializer='glorot_normal'))
# model.add(BatchNormalization())
# model.add(LeakyReLU(alpha=0.2))
# model.add(MaxPooling2D((2,2)))
model.add(AveragePooling2D(pool_size=(3,3)))
model.add(Dropout(0.1)) 
  
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
model.add(AveragePooling2D(pool_size=(3,3))) 
model.add(Dropout(0.1)) 

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
model.add(Dropout(0.1)) 

# model.add(DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='linear'))
# model.add(BatchNormalization())
# model.add(LeakyReLU(alpha=0.2))
# model.add(Conv2D(256,(1,1), activation = 'linear', padding='same',kernel_initializer='glorot_normal'))
# model.add(BatchNormalization())
# model.add(LeakyReLU(alpha=0.2))
# model.add(DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='linear'))
# model.add(BatchNormalization())
# model.add(LeakyReLU(alpha=0.2))
# model.add(Conv2D(256,(1,1), activation = 'linear', padding='same',kernel_initializer='glorot_normal'))
# model.add(BatchNormalization())
# model.add(LeakyReLU(alpha=0.2))
# model.add(MaxPooling2D((2,2)))
 




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
model.add(Dropout(0.1)) 

# model.add(Dense(512,kernel_initializer='glorot_normal'))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5)) 



model.add(Dense(units=7,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=[0.8,1.2],
    shear_range=0.2,
    horizontal_flip=True)

model.summary()


# mcp = keras.callbacks.ModelCheckpoint('mcp-best-acc-{val_acc:.5f}.h5',
#     monitor='val_acc',
#     save_best_only=True,
#     verbose=1,
#     mode='auto',
#     period=1)
# 
# es = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
#     factor=0.1,
#     patience=10,
#     verbose=1,
#     mode='auto',
#     min_delta=0.000001,
#     min_lr=0.0001)
# 
csv_logger = CSVLogger('log.csv', append=False)
learning_rate = ReduceLROnPlateau(monitor='val_acc',factor = 0.2, patience=5, verbose=1, mode='auto', min_delta=1e-4,cooldown=0, min_lr=1e-8)
checkpoint = ModelCheckpoint(filepath='best.h5', monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=True,mode='auto',period=1)
early_stop = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='auto',min_delta=0.0001 )


# datagen.fit(train_feature)
model.fit_generator(datagen.flow(train_feature,train_label,batch_size=256),
    steps_per_epoch=len(train_feature)/4,
    epochs=70,
    validation_data=(valid_feature,valid_label),
    callbacks=[csv_logger,learning_rate,checkpoint,early_stop])
# model.fit(feature,label,batch_size=10,epochs=50)
score = model.evaluate(valid_feature,valid_label)
print('Total loss on testing set : ',score[0])
print('accuracy of testing set : ',score[1])
print('done')


# read testing data #######################################################################
#test_file = sys.argv[2] 
#test_in = parsingTestingData(test_file)

# predict for testing data ################################################################
#result = model.predict(test_in)


# write result ###########################################################################
#writeResult('default.csv',result)


  
# save model
# model.save('best_m.h5')
 
# save history of acc loss
# np_val_acc = np.array(history.history['val_acc'])
# np_tra_acc = np.array(history.history['acc'])
# np_val_loss = np.array(history.history['val_loss'])
# np_tra_loss = np.array(history.history['loss'])
# np.save('best_val_loss',np_val_loss)
# np.save('best_tra_loss',np_tra_loss)
# np.save('best_val_acc',np_val_acc)
# np.save('best_tra_acc',np_tra_acc)

