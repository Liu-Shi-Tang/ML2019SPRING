import sys
import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K


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


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

# read model
model_name = 'mcp-best-acc-0.68250.h5'
model = load_model(model_name)
# read data (label is not on-hot format)
dataFile = sys.argv[1]
outputPath = sys.argv[2]
# For toleration
try :
  os.makedirs('{}'.format(outputPath))
  print('create dir: {}'.format(outputPath))
except:
  print('{} exists!'.format(outputPath))

# loading data from previous 
features, labels =   parsingInput(dataFile)
  
n_classes = 7
# which picture we want to generate saliency map for 
which_picture = [0,1,2,3,4,5,6]

num_steps = 300
record_freq = 50

layer_dict = dict([layer.name, layer] for layer in model.layers)
input_img = model.input

name_ls = ['conv2d_1']

collect_layers = [ layer_dict[name].output for name in name_ls ]

n_filter = 32

for cnt, c in enumerate(collect_layers):
    imgs = []
    for filter_idx in range(n_filter):
        input_img_data = np.random.random((1, 48, 48, 1)) # random noise
        target = K.mean(c[:, :, :, filter_idx])
        grads = normalize(K.gradients(target, input_img)[0])
        iterate = K.function([input_img, K.learning_phase()], [target, grads])

        # calculate img
        input_image_data = np.copy(input_img_data)
        learning_rate = 0.05
        for i in range(num_steps):
            target, grads_val = iterate([input_image_data, 0])
            input_image_data += grads_val * learning_rate
        imgs.append(input_image_data)

    fig , ax = plt.subplots( nrows= n_filter//8, ncols = 8 , figsize=(14, 8))
    plt.tight_layout()
    for j in range(n_filter) :
        origin_img = imgs[j][0].reshape(48, 48, 1)
        origin_img = np.clip(origin_img, 0, 1).astype('uint8')
        ax[j//8,j%8].imshow(origin_img.squeeze(), cmap='Oranges')
        ax[j//8,j%8].set_title('filter {}'.format(j)) 
    fig.savefig('{}fig2_1.jpg'.format(outputPath))
    plt.close()

