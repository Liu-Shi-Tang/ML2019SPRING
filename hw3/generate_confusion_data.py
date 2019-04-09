import numpy as np
import sys

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

  np.save('feature_valid',valid_feature)
  np.save('label_valid',valid_label)
  

parsingTrainingData(sys.argv[1])

print('done')
