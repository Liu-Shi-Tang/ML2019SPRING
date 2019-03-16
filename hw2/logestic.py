import numpy as np
import sys
from sklearn.model_selection import train_test_split


raw_train = sys.argv[1]
raw_test = sys.argv[2]
train_feature = sys.argv[3]
train_label   = sys.argv[4]
test_feature  = sys.argv[5]
prediction_file = sys.argv[6]

# read in parsed data for training #######################################################################
data_in = np.genfromtxt(train_feature,delimiter = ',',skip_header=1) 
data_in = np.concatenate((data_in,data_in*np.log(data_in+1e-8),np.log(data_in+1e-8)),axis=1)
label_in = np.genfromtxt(train_label,delimiter = ',' , skip_header=1)
# print (np.shape(data_in),np.shape(label_in))

# normalize data #######################################################################################
std = np.std(data_in,axis=0)
mean = np.mean(data_in,axis=0)
data_in = np.divide(np.subtract(data_in,mean),std)


data_in,test_x,label_in,test_y = train_test_split(data_in,label_in,test_size=0.3,random_state=0)


# define sigmoid function for logistic regression #######################################################
def mysigmoid(z) :
  return np.clip(1 / ( 1 + np.exp(-1*z)),0.00000000000000001,0.999999999999999)


# training ##############################################################################################
num_fea = len(data_in[0])
num_data = data_in.shape[0]
data_tran = data_in.transpose()
lr = 0.1
iteration = 3000
w = np.zeros(num_fea)
# print(len(data_in[0]))
b = 0.1
s_w = np.zeros(num_fea)
s_b = 0.0

loss_test = []
loss_train = []

lamda = 0

for it in range(iteration) :
  # TO-DO
  pre = mysigmoid(np.dot(data_in,w)+b)
  dif = label_in - pre
  wg = -1 * np.dot(data_tran,dif) 
  bg = -1 * np.sum(dif) * 1 
  s_w += wg**2
  s_b += bg**2
  
  w -= lr*wg/np.sqrt(s_w) + lamda*w
  b -= lr*bg/np.sqrt(s_b)


  if it%10 == 0 : 
    t_pre = (pre >= 0.5)
    t_dif = (t_pre == label_in ) 
    t_dif = t_dif.astype('int')
    l = np.average(t_dif)
    loss_train.append(l)

    pre = mysigmoid(np.dot(test_x,w)+b)
    t_pre = (pre >= 0.5)
    t_dif = (t_pre == test_y ) 
    t_dif = t_dif.astype('int')
    l = np.average(t_dif)
    loss_test.append(l)


import matplotlib.pyplot as plt
plt.plot(loss_test,c='r')
plt.plot(loss_train,c='b')
plt.show()


test_data = np.genfromtxt(test_feature,delimiter = ',',skip_header=1)
test_data = np.concatenate((test_data,test_data*np.log(test_data + 1e-8),np.log(test_data + 1e-8)),axis=1)

# normalize data #######################################################################################
test_std = np.std(test_data,axis=0)
test_std += 0.00000000000000000001
test_mean = np.mean(test_data,axis=0)
test_data = np.divide(np.subtract(test_data,test_mean),std)


# print(np.shape(test_data))
out = open(prediction_file,'w')
y_test = ( mysigmoid(np.dot(test_data,w)+b) >= 0.5 )
# print(len(y_test))
out.write("id,label\n")
for i in range(len(y_test)) :
  out.write(str(i+1) + ',' + str(int(y_test[i])) + '\n')
out.close()


