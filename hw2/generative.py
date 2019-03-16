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
label_in = np.genfromtxt(train_label,delimiter = ',' , skip_header=1)
# print (np.shape(data_in),np.shape(label_in))

# Find Mu for two class ####################################################################################

size_of_feature = len(data_in[0])


num_of_class_high = 0 
num_of_class_low  = 0

mean_high = np.zeros((1,size_of_feature))
mean_low  = np.zeros((1,size_of_feature))

# We should divide data into two group
class_high = []
class_low  = []

for i in range(len(data_in)) :
  if label_in[i] == 1 :
    class_high.append(data_in[i])
    mean_high += data_in[i] 
    num_of_class_high += 1
  else :
    class_low.append(data_in[i])
    mean_low += data_in[i]
    num_of_class_low += 1


if num_of_class_high > 0 :
  mean_high /= num_of_class_high
if num_of_class_low > 0 :
  mean_low /= num_of_class_low


# start to calculate sigma for same covariance

print(np.transpose((class_high[0] - mean_high)).shape,type((class_high[0] - mean_high)))
print(np.dot( np.transpose(class_low[0]  - mean_low ) , class_low[0] - mean_low  ).shape)


cov_high = np.zeros((size_of_feature,size_of_feature))
cov_low  = np.zeros((size_of_feature,size_of_feature))

for i in range(num_of_class_high) :
  cov_high += np.dot( np.transpose(class_high[i]  - mean_high ) , class_high[i] - mean_high  ) / num_of_class_high
for i in range(num_of_class_low) :
  cov_low += np.dot( np.transpose(class_low[i]  - mean_low ) , class_low[i] - mean_low  ) / num_of_class_low



cov = (cov_high*num_of_class_high + cov_low*num_of_class_low ) / (num_of_class_high + num_of_class_low)

print(cov.shape,type(cov))
cov_inv = np.linalg.inv(cov)

w = np.dot(mean_high-mean_low,cov_inv)
b = (-0.5) * np.dot(np.dot(mean_high,cov_inv) , np.transpose(mean_high) ) + 0.5 * np.dot( np.dot(mean_low,cov_inv), np.transpose(mean_low) ) + np.log(float(num_of_class_high)/num_of_class_low)


# read testing data
test_data = np.genfromtxt(test_feature,delimiter = ',',skip_header=1)



z = np.dot(test_data,np.transpose(w)) + b
ans = 1 / ( 1 + np.exp(-z))
ans = (ans >= 0.5)


out = open(prediction_file,'w')
# print(len(y_test))
out.write("id,label\n")
for i in range(len(ans)) :
  out.write(str(i+1) + ',' + str(int(ans[i])) + '\n')
out.close()


