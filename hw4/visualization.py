import sys
# import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
# from utils import *
# from marcos import *

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
outputPath = sys.argv[2]
# loading data from previous 
features, labels =   parsingInput(dataFile)
  
n_classes = 7
# which picture we want to generate saliency map for 
which_picture = [0,1,2,3,4,5,6]


# input layer
input_img = model.input
# Get layers for observation
focus_layers = []
for layer in model.layers :
    print(layer.name)
    if layer.name == 'conv2d_1' :
        focus_layers.append( K.function([input_img , K.learning_phase()] , [layer.output]) )


for i,layer in enumerate(focus_layers)  :
    # randomly choosing a picture from 7 picture (size = 48x48x1)
    picture = features[0].reshape(1,48,48,1)
    output = layer([picture,0])
    # number of ofilter for observation
    n_filter = 32
    fig , ax = plt.subplots( nrows= n_filter//8, ncols = 8 , figsize=(14, 8))
    plt.tight_layout()
    for j in range(n_filter) :
        # ax = fig.add_subplot(n_filter/8, 8, i+1)
        ax[j//8,j%8].imshow(output[0][0, :, :, j], cmap='Oranges')
        ax[j//8,j%8].set_title('filter {}'.format(j)) 
    fig.savefig('{}fig2_2.jpg'.format(outputPath))
    plt.close()


