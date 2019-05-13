from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from PIL import Image
import os
import sys



def getModel (shape_in=(32,32,3)) :
	input_img = Input(shape=shape_in)
	x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
	x = MaxPooling2D((2, 2), padding='same')(x) 								# width,height /= 2
	x = Conv2D(3, (3, 3), activation='relu', padding='same')(x)
	encoded = MaxPooling2D((2, 2), padding='same')(x) 							# width,height /= 2

	# shape of encode is (32/4,32/4,3)

	x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded) 
	x = UpSampling2D((2, 2))(x)													# width,height *= 2
	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)													# width,height *= 2
	decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

	# shape of decode is (32,32,3)

	autoencoder = Model(input_img,decoded)

	return encoded,decoded,autoencoder

#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

inputDir = sys.argv[1] 

# extract file name
pngList = [ file for file in os.listdir(inputDir) if file.endswith('.jpg')]

pictures = [ np.asarray(Image.open(pic)) for pic in pngList ]
pictures = np.array(pictures)

print()

