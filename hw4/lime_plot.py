import numpy as np
from lime import lime_image
from skimage.segmentation import slic
import matplotlib.pyplot as plt
import keras
from keras.models import Model, load_model
# for segmentation function
from lime.wrappers.scikit_image import SegmentationAlgorithm
# for masking boundary ???
from skimage.segmentation import mark_boundaries
import skimage


# number of class
num_class = 7 

# loading previous model
model_name = 'mcp-best-acc-0.68250.h5'
model = load_model(model_name)

# loading data for explanation
features = np.load('features.npy')
labels = np.load('labels.npy')
# features = features.reshape(-1,48,48)
# features = features.reshape(-1,48,48,1)
print(features.shape)

def predicFunction(features) :
    # features = features.reshape(48,48,1)
    return model.predict(features)
    # r = model.predict(features)
    # return np.argmax(r)

def segmentation(features):
    np.random.seed(666)
    return skimage.segmentation.slic(
        image=features,
        n_segments=250)

index = 5
# def returnFixSeedSegment_fun(feature) :
#     np.random.seed(666)
#     return skimage.segmentation.slic(feature,n_segments=4,compactness=20)

# def explain(instance, predict_fn, **kwargs):
#   np.random.seed(16)
#   return exp.explain_instance(instance, predict_fn, **kwargs)

# Initiate explainer instance
explainer = lime_image.LimeImageExplainer()

# Get the explaination of an image
explanation = explainer.explain_instance(
    image=features[index], 
    classifier_fn=predicFunction,
    segmentation_fn=segmentation,
    hide_color = 1,
    num_samples=250
    )


# Get processed image
image, mask = explanation.get_image_and_mask(
    label=labels[index],
    positive_only=True,
    hide_rest=False,
    num_features=30,
    min_weight=0.0)

# save the image
# print (image)
# print (features[0])
# print (image-features[2])
# print (image.shape,mask.shape)

x = image
x = x.reshape(48,48)
# x *= 255
# x *= mask
print(np.max(x))
# x = skimage.color.gray2rgb(x)
# print(type(x) )
# print(x.shape)
# print(mask)
# x = features[2].reshape(48,48)
# plt.imshow(x,cmap='gray',interpolation ='nearest')
plt.imshow(mark_boundaries(x, mask),cmap='gray',interpolation ='nearest')
plt.savefig("img")
plt.show()
# plt.imshow(x, cmap='gray', vmin=0, vmax=255)
# plt.imsave('img', x)

