import sys
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Model, load_model

# TO limit memory usage for protability
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config_tf = tf.ConfigProto()
# config_tf.gpu_options.per_process_gpu_memory_fraction = 0.7
# sess = tf.Session(config=config_tf)
# set_session(sess)


def parsingInput(file_name) :
  # size : 48 X 48
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
  # label = np_utils.to_categorical(label,7)

  # reshaping for convolution
  feature = np.reshape(feature,(num_train,48,48,1))

  # np.save('train_X.npy', feature)
  # sys.exit(0)

  feature = feature.astype(float)
  feature = feature/255

  sal_feature = []
  sal_label = []
  for i in range(7) :
    for j in range(len(feature)) :
      if ( int(label[j]) == int(i) ) :
        sal_feature.append(feature[j])
        sal_label.append(label[j])
        print(j)
        break ;
  sal_feature = np.array(sal_feature)
  sal_label = np.array(sal_label)
  return sal_feature,sal_label


# read model
model_name = 'mcp-best-acc-0.68250.h5'
model = load_model(model_name)
# read data (label is not on-hot format)
dataFile = sys.argv[1]
# loading data from previous 
features, labels =   parsingInput(dataFile)
  
n_classes = 7
# which picture we want to generate saliency map for 
which_picture = [0,1,2,3,4,5,6]



for i in which_picture :
  # get a image
  img = features[i]
  print(labels[i])
 
  # To get the gradient
  gradients = model.optimizer.get_gradients(model.output[0][int(labels[i])], model.input)
  # To create an keras object
  compute_gradients = K.function(inputs = [model.input], outputs = gradients)
  # Execute the function to compute the gradient
  img_exp = np.expand_dims(img, axis=0)

  # To get the saliency map
  sal_map = compute_gradients([img_exp])[0][0]
  sal_map = np.abs(sal_map) 
  ma = np.max(sal_map)
  # For masking partial figure
  filter_map = (sal_map > (ma*0.08)).reshape((48, 48))

  # start to ploting
  fig, ax = plt.subplots(nrows=1,  ncols=3)
  fig.suptitle('SaliencyMap {}'.format(labels[i]))
  ax[0].imshow(img.reshape((48, 48)), cmap = 'gray')
  cax = ax[1].imshow(sal_map.reshape((48, 48)), cmap = 'jet')
  fig.colorbar(cax, ax = ax[1])
  ax[2].imshow(img.reshape((48, 48)) * filter_map, cmap = 'gray')
  plt.savefig('fig1_{}.jpg'.format(int(labels[i])))

print('Finish saliency map')



