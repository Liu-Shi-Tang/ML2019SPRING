import numpy as np
import pandas as pd
import sys

arg_test = sys.argv[1]
arg_output = sys.argv[2]


# loading model 
w = np.load("model/train.npy")

test_in = pd.read_csv(arg_test,encoding='Big5',header=None).iloc[:,2:]
x_test = []
for i in range(240) :
    x_test.append([])
    for j in range(18) :
        for k in range(4,9) :
            if test_in.iloc[i*18+j,k] != 'NR' :
                x_test[i].append(float(test_in.iloc[i*18+j,k]))
            else :
                x_test[i].append(float(0))
    x_test[i].append(float(1))

x_test = np.array(x_test)
# print(np.shape(x_test))
y_pre = np.dot(x_test,w)
outputFile = open(arg_output,'w+')
outputFile.write("id,value\n")
for i in range(len(y_pre)) :
    outputFile.write("id_"+str(i)+","+str(y_pre[i])+"\n")
outputFile.close()
