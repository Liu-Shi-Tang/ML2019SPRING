import sys
import csv
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def parsingTrainingData ( file_name ) :
  # read data
  # please note that the first element in each line have label
  # The reason why I use str type is that the first element of each row is str type
  data_in = np.genfromtxt(fname=file_name,skip_header=1,dtype=str,delimiter=' ')
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

  return train_feature , train_label , valid_feature , valid_label




def readTestingData( file_name ) :
  # read testing data
  test_in = np.genfromtxt(fname=file_name,skip_header=1,dtype=str,delimiter=' ')
  num_test = len(test_in)
  
  for i in range(num_test) :
    test_in[i,0] = test_in[i,0].split(',')[1]
  
  
  # reshaping testing data
  test_in = np.reshape(test_in,(num_test,48,48,1))
  test_in = test_in.astype(float)
  test_in = test_in/255

  return test_in


def writeResult(result,file_name='default.csv') :
  f = open(file_name,'w')
  f.write("id,label\n")
  for i in range(len(result)) :
    f.write(str(i) + ',' + str(int(np.argmax(result[i]))) + '\n')
  
  f.close()
  print('writeResult finish!')


# define my network
class CNN01 (nn.Module):
  def __init__ (self) :
    super(CNN01,self).__init__()
    


# arg
training_data_file = sys[1]
testing_data_file = sys[2]


# read training data
train_feature , train_label , valid_feature , valid_label = parsingTrainingData(training_data_file)




