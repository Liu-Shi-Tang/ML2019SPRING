import numpy as np
import sys

raw_train = sys.argv[1]
raw_test = sys.argv[2]
train_feature = sys.argv[3]
train_label   = sys.argv[4]
test_feature  = sys.argv[5]
prediction_file = sys.argv[6]

# read in parsed data for training #######################################################################
data_in = np.genfromtxt(train_feature,delimiter = ',',skip_header=1) 
label_in = np.genfromtxt(train_label,delimiter = ',' , skip_header=1)
# print (np.shape(data_in),np.shape(label_in))


# define sigmoid function for logistic regression #######################################################
def mysigmoid(z) :
  return np.clip(1 / ( 1 + np.exp(-1*z)),0.00000000000000001,0.999999999999999)


# training ##############################################################################################
num_fea = len(data_in[0])
num_data = data_in.shape[0]
data_tran = data_in.transpose()
lr = 0.1
iteration = 10000
w = np.zeros(num_fea)
# print(len(data_in[0]))
b = 0
s_w = np.zeros(num_fea)
s_b = 0.0

loss = []


for it in range(iteration) :
  # TO-DO
  pre = mysigmoid(np.dot(data_in,w)+b)
  dif = label_in - pre
  wg = -1 * np.dot(data_tran,dif) / num_data
  bg = -1 * np.sum(dif) * 1 / num_data
  s_w += wg**2
  s_b += bg**2
  
  w += lr*wg/np.sqrt(s_w)
  b += lr*bg/np.sqrt(s_b)
 
  loss.append( -1 * (   label_in*np.log(pre) + (1-label_in)*np.log(1-pre) )    ) 

import matplotlib.pyplot as plt
plt.plot(loss[1000:])
plt.show()
