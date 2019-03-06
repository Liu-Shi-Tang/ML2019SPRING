import numpy as np
import pandas as pd
import sys

# Read data ***********************************************************************
arg_input  = sys.argv[1]
data_in = pd.read_csv(arg_input,encoding='Big5').iloc[:,3:]
num_row = data_in.shape[0]
num_col = data_in.shape[1]

# print(num_row,num_col)
# print(data_in.iloc[2,3])

# reshape data to num X 18 ********************************************************
data = []
for i in range(18) :
    data.append([])

for n_r in range(num_row) :
    for n_c in range(num_col) :
        if data_in.iloc[n_r,n_c] != 'NR' :
            data[n_r%18].append(float(data_in.iloc[n_r,n_c]))
        else :
            data[n_r%18].append(float(0))

# Parsing *************************************************************************
x_train = []
y_train = []

for m in range(12) :
    # 20*24 - 9 = 471
    for h in range(471) :
        x_train.append([])
        # n_f = num of feature
        for n_f in range(18) :
            # n_e = num of elements
            for n_e in range(9) :
                x_train[m*471+h].append(data[n_f][480*m+h+n_e])
        # bias
        x_train[m*471+h].append(1)
        y_train.append(data[9][480*m+h+9])

x_train = np.array(x_train)
y_train = np.array(y_train)


# print(np.shape(x_train))

# Training data ******************************************************************
l_rate  = 0.1
n_ite   = 90000
x_tp    = x_train.transpose()
w       = np.zeros(len(x_train[0]))
loss = []

# adam  
beta_1 = 0.8
beta_2 = 0.9
epsilon = 1e-10
mt = np.zeros(len(w))
vt = np.zeros(len(w))

for it in range(n_ite) :
    diff = y_train - np.dot(x_train,w)
    gra = 2.0 * np.dot(x_tp,diff) * (-1) / x_tp.shape[1]
    mt = beta_1*mt + (1-beta_1)*gra
    vt = beta_1*vt + (1-beta_2)*(gra**2)
    w -= l_rate*(mt/(1-beta_1))/(np.sqrt(vt/(1-beta_2))+ epsilon)
    loss.append(np.sqrt(np.dot(diff,diff)/len(diff)))

np.save("model/train",w)
