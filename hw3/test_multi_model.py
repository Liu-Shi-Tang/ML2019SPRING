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

def writeFile(result_file,result_in) :
  f = open(result_file,'w')
  f.write("id,label\n")
  for i in range(len(result_in)) :
    f.write(str(i) + ',' + str(int(np.argmax(result_in[i]))) + '\n')
  f.close()
 


# read testing data
test_file = sys.argv[1] 
test_in = np.genfromtxt(fname=test_file,skip_header=1,dtype=str,delimiter=' ')
num_test = len(test_in)

for i in range(num_test) :
  test_in[i,0] = test_in[i,0].split(',')[1]

# reshaping testing data
test_in = np.reshape(test_in,(num_test,48,48,1))
test_in = test_in.astype(float)


# load model
model1 = load_model('mcp-no-all-acc-0.63350.h5')
model2 = load_model('mcp-nor-only-acc-0.64050.h5')
model3 = load_model('mcp-both-acc-0.68450.h5')
model4 = load_model('mcp-dnn-acc-0.46650.h5')
mean = np.load('both_mean.npy')
std = np.load('both_std.npy')

# without std
result1 = model1.predict(test_in)
test_in = test_in/255


# using std
test_in = (test_in - mean) / (std + 1e-20)  
result2 = model2.predict(test_in)
result3 = model3.predict(test_in)
test_in = test_in.reshape(-1,48*48)
result4 = model4.predict(test_in)

# write result
result_file_1 = 'no_all.csv'
result_file_2 = 'nor_only.csv'
result_file_3 = 'both.csv'
result_file_4 = 'dnn_nor.csv'

writeFile(result_file_1,result1)
writeFile(result_file_2,result2)
writeFile(result_file_3,result3)
writeFile(result_file_4,result4)
print('end')


