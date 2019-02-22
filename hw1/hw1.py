import numpy as np
import pandas as pd
import sys

# Read data ***********************************************************************
arg_input  = sys.argv[1]
arg_output = sys.argv[2]
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

# Observe data relationship for selecting model

import matplotlib.pyplot as plt

names   = pd.read_csv(arg_input,encoding='Big5').iloc[:18,2]
colors  = [(i/3.1,j/3.1,k/3.1) for i in range(3) for j in range(3) for k in range(2) ]
size    = 20
for i in range(18) :
    plt.scatter(x_train[i],x_train[9],s=size,color=colors[i],label=names[i])
plt.legend()
plt.show()
