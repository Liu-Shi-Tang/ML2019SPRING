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

model1 = load_model( './model/mcp-20190407-1-acc-0.67350.h5' )
model2 = load_model( './model/mcp-20190407-3-acc-0.68950.h5' )

result1 = model1.predict(test_in)
result2 = model2.predict(test_in)
result = result1 + result2

f = open('default.csv','w')
f.write("id,label\n")
for i in range(len(result)) :
  f.write(str(i) + ',' + str(int(np.argmax(result[i]))) + '\n')

f.close()
  
print('end')


