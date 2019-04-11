import numpy as np
from lime import lime_image
from skimage.segmentation import slic
import matplotlib.pyplot as plt
import keras
from keras.models import Model, load_model
# for segmentation function
# from lime.wrappers.scikit_image import SegmentationAlgorithm
# for masking boundary ???
from skimage.segmentation import mark_boundaries
import skimage


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

############################################# start to run lime ########################################
def predicFunction(features) :
    features = skimage.color.rgb2gray(features)
    features = features.reshape(-1,48,48,1)
    return model.predict(features)

def segmentation(features):
    np.random.seed(0)
    return skimage.segmentation.slic(
        image=features)


for i in range(7) :
  # size of features is (7,48,48,1)
  in_picture = features[i]
  in_picture = np.reshape(in_picture,(48,48))
  in_picture = skimage.color.gray2rgb(in_picture)
  
  # Initiate explainer instance
  explainer = lime_image.LimeImageExplainer()
  
  # Get the explaination of an image
  explanation = explainer.explain_instance(
      image=in_picture, 
      classifier_fn=predicFunction,
      segmentation_fn=segmentation,
      hide_color = 0,
      num_samples=4000,
      top_labels=10
      )
  
  
  # Get processed image
  image, mask = explanation.get_image_and_mask(
      label=labels[i],
      positive_only=False,
      hide_rest=False,
      num_features=10,
      min_weight=0.0)
  
    
  x = image
  x = x.reshape(48,48,3)
  plt.imshow(mark_boundaries(skimage.color.gray2rgb(x), mask),interpolation ='nearest')
  plt.savefig("fig3_{}.jpg".format(int(labels[i])))

