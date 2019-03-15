import numpy as np
import sys
from sklearn.model_selection import train_test_split


raw_train = sys.argv[1]
raw_test = sys.argv[2]
train_feature = sys.argv[3]
train_label   = sys.argv[4]
test_feature  = sys.argv[5]
prediction_file = sys.argv[6]

# read in parsed data for training #########################################################################
data_in = np.genfromtxt(train_feature,delimiter = ',',skip_header=1) 
data_in = np.concatenate((data_in,data_in**2,data_in**3,data_in**4,np.log(data_in + 1e-8),data_in*np.log(data_in + 1e-8)),axis=1)
label_in = np.genfromtxt(train_label,delimiter = ',' , skip_header=1)
# print (np.shape(data_in),np.shape(label_in))

# define sigmoid function for prediction ###################################################################
def mysigmoid(z) :
  return np.clip(1 / ( 1 + np.exp(-1*z)),0.00000000000000001,0.999999999999999)

# Find Mu for two class ####################################################################################

size_of_feature = len(data_in[0])


num_of_class_high = 0 
num_of_class_low  = 0

mean_high = np.zeros((1,size_of_feature))
mean_low  = np.zeros((1,size_of_feature))


for i in range(len(data_in)) :
  if label_in[i] == 1 :
    mean_high += data_in[i] 
    num_of_class_high += 1
  else :
    mean_low += data_in[i]
    num_of_class_low += 1


if num_of_class_high > 0 :
  mean_high /= num_of_class_high
if num_of_class_low > 0 :
  mean_low /= num_of_class_low



