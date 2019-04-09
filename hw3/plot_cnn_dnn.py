import matplotlib.pyplot as plt 
import numpy as np

loss_cnn = np.load('./hw3_model/new_model/both_tra_loss.npy')
loss_dnn = np.load('./hw3_model/new_model/dnn_tra_loss.npy')
acc_cnn = np.load('./hw3_model/new_model/both_tra_acc.npy')
acc_dnn = np.load('./hw3_model/new_model/dnn_tra_acc.npy')



# plt.plot(loss_cnn,c='r',label='loss_cnn')
# plt.plot(loss_dnn,c='g',label='loss_dnn')
# plt.legend()
# plt.show()

plt.plot(acc_cnn,c='r',label='accuracy_cnn')
plt.plot(acc_dnn,c='g',label='accuracy_dnn')
plt.legend()
plt.show()