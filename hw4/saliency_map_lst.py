# import os
import sys
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Model, load_model

# TO limit memory usage for protability
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config_tf = tf.ConfigProto()
config_tf.gpu_options.per_process_gpu_memory_fraction = 0.7
sess = tf.Session(config=config_tf)
set_session(sess)
# sess.run(init)
# base_dir = './'
# img_dir = os.path.join(base_dir, 'image')
# if not os.path.exists(img_dir):
#   os.makedirs(img_dir)
# cmap_dir = os.path.join(img_dir, 'cmap')
# if not os.path.exists(cmap_dir):
#   os.makedirs(cmap_dir)
# partial_see_dir = os.path.join(img_dir,'partial_see')
# if not os.path.exists(partial_see_dir):
#   os.makedirs(partial_see_dir)
# origin_dir = os.path.join(img_dir,'origin')
# if not os.path.exists(origin_dir):
#   os.makedirs(origin_dir)

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
  # np.save('Y_train_label',sal_label)
  # np.save('X_train',sal_feature)
  return sal_feature,sal_label

class GradientSaliency():
  def __init__(self, model, output_index = 0):
    '''
      define a function to return gradient from specified output
      output_index  : if the label of a picture is 6 , then we will calaulate the gradient
                      from input to the 6-th output
      model         : the model we trained before
    '''
    # Define the function to compute the gradient

    # To get the input layer, and it will be used in calculating gradient
    input_tensors = [model.input]
    # To get the gradient
    gradients = model.optimizer.get_gradients(model.output[0][output_index], model.input)
    # To create an keras object
    self.compute_gradients = K.function(inputs = input_tensors, outputs = gradients)

  def get_mask(self, input_image):
    # Execute the function to compute the gradient
    x_value = np.expand_dims(input_image, axis=0)
    gradients = self.compute_gradients([x_value])[0][0]
    # return the gradient calculated by keras object which was created in constructor
    return gradients


# read model
model_name = 'mcp-best-acc-0.68250.h5'
model = load_model(model_name)
# read data (label is not on-hot format)
dataFile = sys.argv[1]
# loading data from previous 
features, labels =   parsingInput(dataFile)
print(features.shape,labels.shape)
  
n_classes = 7
# which picture we want to generate saliency map for 
which_picture = [0,1,2,3,4,5,6]

# X_train = np.load('X_train.npy')
# Y_train_label = np.load('Y_train_label.npy')


fig, ax = plt.subplots(nrows=n_classes,  ncols=3)
fig.suptitle('SaliencyMap')



for i in which_picture :
  # get a image
  img = features[i]
  print(labels[i])
  # initialize customized object
  vanilla = GradientSaliency(model, int(labels[i]))
  # To get the mask from gradients
  mask = vanilla.get_mask(img)
  filter_mask = (mask > 0.0).reshape((48, 48))


  ax[i, 0].imshow(img.reshape((48, 48)), cmap = 'gray')
  cax = ax[i, 1].imshow(mask.reshape((48, 48)), cmap = 'jet')
  fig.colorbar(cax, ax = ax[i, 1])
  ax[i, 2].imshow(mask.reshape((48, 48)) * filter_mask, cmap = 'gray')
plt.savefig('Saliency_map.png')
plt.show()
