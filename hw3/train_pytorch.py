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
  feature = np.reshape(feature,(num_train,1,48,48)) # different from keras
  
  # np.save('train_X.npy', feature)
  # sys.exit(0)
  
  feature = feature.astype(float)
  feature = feature/255
  
  
  # split data
  valid_feature = feature[:2000]
  valid_label = label[:2000]
  train_feature = feature[2000:]
  train_label = label[2000:]

  # Transform to tensor data
  valid_feature = torch.FloatTensor(valid_feature)
  train_feature = torch.FloatTensor(train_feature)
  valid_label   = torch.LongTensor(valid_label)
  train_label   = torch.LongTensor(train_label)
  return train_feature , train_label , valid_feature , valid_label




def readTestingData( file_name ) :
  # read testing data
  test_in = np.genfromtxt(fname=file_name,skip_header=1,dtype=str,delimiter=' ')
  num_test = len(test_in)
  
  for i in range(num_test) :
    test_in[i,0] = test_in[i,0].split(',')[1]
  
  
  # reshaping testing data
  test_in = np.reshape(test_in,(num_test,1,48,48)) # different from keras
  test_in = test_in.astype(float)
  test_in = test_in/255
 
  # Transform to tensor data 
  test_in = torch.FloatTensor(test_in)

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
    self.cnn = nn.Sequential(
        nn.Conv2d(1,64,5,1,2),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2),
        nn.Conv2d(64,64,3,1,1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2),
        nn.MaxPoolsd(2,2,0),

        nn.Conv2d(64,128,3,1,1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2),
        nn.Conv2d(128,128,3,1,1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2),
        nn.MaxPoolsd(2,2,0),

        nn.Conv2d(128,256,3,1,1),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2),
        nn.Conv2d(256,256,3,1,1),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2),
        nn.MaxPoolsd(2,2,0),
        )

    self.fc = nn.Sequential(
        nn.Linear(256*6*6,1024),
        nn.LeakyReLU(0.2),
        nn.Dropout(p=0.5),
        nn.Linear(1024,512),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.5),
        nn.Softmax()
        )

  def forward(self,x) :
    out = self.cnn(x)
    out = out.view(out.size()[0],-1) # flatten
    return self.fc(out)

# arg
training_data_file = sys[1]
testing_data_file = sys[2]


# read training data
train_feature , train_label , valid_feature , valid_label = parsingTrainingData(training_data_file)


# convert to TensorDataSet
train_set = TensorDataset(train_feature,train_label)
val_set   = TensorDataset(valid_feature,valid_label)

# define parameters
BATCH_SIZE = 256
LEARNING_RATE = 0.01
EPOCHES = 20

# convert to DataLoader
train_loader = DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=False,num_worker=8)
val_loader   = DataLoader(val_set,batch_size=BATCH_SIZE,shuffle=False,num_worker=8)

# training
model = CNN01().cuda()

# print model
print(model)

loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
best_acc = 0.0

for epoch in range(EPOCHES) :
  epoch_start_time = time.time()
  train_acc = 0.0
  train_loss= 0.0
  val_acc   = 0.0
  val_loss  = 0.0

  # start for training
  model.train()
  for i,data in enumerate(train_loader) :
    optimizer.zero_grad()

    train_pre = model(data[0].cuda())
    batch_loss= loss_fun(train_pre,data[1].cuda())
    batch_loss.backward()
    optimizer.step()

    train_acc += np.sum





 

