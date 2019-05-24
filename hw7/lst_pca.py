import numpy as np
from numpy.linalg import svd
from skimage import io
import sys
import os

# directory with images
IMAGES_PATH = sys.argv[1] 
# name of image
INPUT_IMAGE = sys.argv[2] 
RECONSTRUCT_IMAGE = sys.argv[3] 


# Number of principal components used
k = 5
# Number of testing pictures
NUM_OF_PICTURE = 415

# TA code for process image
def process(M): 
    M = np.copy(M)
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    return M

# For 1.a
def averageFace(images):
    # Save average.jpg
    average = np.mean(images, axis=0)
    average = process(average)
    io.imsave('average.jpg', average.reshape(600,600,3))

# For 1.b and 1.d
def eigenFace(images) :
    images = images.astype(np.float64)
    average = np.mean(images, axis=0).astype(np.float64)
    images -= average

    # Use SVD to find the eigenvectors 
    eigen, s, v = np.linalg.svd(images.transpose(), full_matrices=False)
    # This eigen have been sorted by their eigen value

    '''
    shape of three var:
      eigen = 1080000*415
      s = 415*1
      v = 415*415
    '''

    # Plot eigenface for 1.b
    for i in range(10):
        eigenface = process(-eigen[:,i]).reshape(600,600,3)
        io.imsave(str(i) + '_eigenface.jpg', eigenface, quality=100)

    # For 1.d 
    for i in range(5):
        percentage = s[i] * 100 / sum(s)
        print(i, ' Eigenfaces: ', round(percentage, 1))

    return(eigen)

# For 1.c
def reconstruction(images, eigen):
    # Load specified picture for reconstrcution
    img = io.imread(os.path.join(IMAGES_PATH, INPUT_IMAGE))
    testImage = img.flatten().astype('float32') 

    # Minus average for compression (important)
    average =  np.mean(images, axis=0)
    testImage -= average
    

    # Compression
    weight = np.dot(testImage, eigen)
    
    # Reconstruction
    reconPic = np.zeros(600*600*3)
    # Only take top k eigen vectors for reconstruction
    for i in range(k):
        projection = weight[i] * eigen[:,i]
        reconPic += projection
    # add average to the reconstructed picture
    reconPic += average
    reconPic = process(reconPic).reshape(600,600,3)
    io.imsave(RECONSTRUCT_IMAGE, reconPic, quality=100)



if __name__ == "__main__":
    
    images = list()
    # Load 415 pictures modify from TA code
    for index in range(NUM_OF_PICTURE):
        picPath = os.path.join(IMAGES_PATH, str(index)+'.jpg')
        img = io.imread(picPath)
        images.append(img.flatten())
    training_data = np.array(images).astype('float32')
    
    # prob 1.a
    averageFace(training_data)
    # prob 1.b 1.d
    eigen = eigenFace(training_data)
    # prob 1.c
    reconstruction(training_data, eigen)
