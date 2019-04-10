import numpy as np
import sys
from keras.applications import vgg16
from keras.preprocessing import image
from keras.activations import relu, softmax
from keras.models import load_model
import keras.backend as K
import scipy.misc
pic_num = 20
start = int(sys.argv[1])
def plot_img(i, x):
    """
    x is a BGR image with shape (? ,224, 224, 3) 
    """
    t = np.zeros_like(x[0])
    t[:,:,0] = x[0][:,:,2]
    t[:,:,1] = x[0][:,:,1]
    t[:,:,2] = x[0][:,:,0]  
    t = np.clip((t+[123.68, 116.779, 103.939]), 0, 255)/255
    scipy.misc.toimage(t).save('attacked_images/{:0>3d}.png'.format(i))

model = vgg16.VGG16(weights='imagenet')
#model = load_model('vgg16.h5')
#img_path = 'images/001.png'
x = np.zeros((pic_num,224,224,3))


for i in range(start,start+pic_num):
    #x = np.zeros((1,224,224,3))
    img = image.load_img('../../ML_data/hw5/images/{:0>3d}.png'.format(i), target_size=(224,224))
    x[i%pic_num] = image.img_to_array(img)
'''
plt.imshow(img)
plt.grid('off')
plt.axis('off')
'''
# Create a batch and preprocess the image
#x = image.img_to_array(img)
x = vgg16.preprocess_input(x)

# Get the initial predictions
preds = model.predict(x)
initial_class = np.argmax(preds,axis=1)

#print('Predicted:', vgg16.decode_predictions(preds, top=3))

# Get current session (assuming tf backend)
sess = K.get_session()
# Initialize adversarial example with input image
x_adv = x
# Added noise
x_noise = np.zeros_like(x)

# Set variables
epochs = 150
epsilon = 0.02
#prev_probs = []
target = K.one_hot(initial_class, 1000)
for j in range(epochs): 
    # One hot encode the initial class
    
    # Get the loss and gradient of the loss wrt the inputs
    loss = K.categorical_crossentropy(target, model.output)
    grads = K.gradients(loss, model.input)

    # Get the sign of the gradient
    delta = K.sign(grads[0])
    x_noise = x_noise + delta

    # Perturb the image
    x_adv = x_adv + epsilon*delta

    # Get the new image and predictions
    x_adv = sess.run(x_adv, feed_dict={model.input:x})
    preds = model.predict(x_adv)

    # Store the probability of the target class
    #prev_probs.append(preds[0][initial_class])

    print('start :',start,' epochs :',j)
     #   print(j, preds[0][initial_class], vgg16.decode_predictions(preds, top=3)[0])

for i in range(start,start+pic_num):
	plot_img(i, np.expand_dims(x_adv[(i%pic_num)], axis=0))
#plot_img(i, np.expand_dims(x_adv[i]-x[i], axis=0))

#plt.plot(np.arange(0,len(prev_probs)), prev_probs)
#plt.show()

#np.save('x', x)
#np.save('x_adv', x_adv)
