
# $1 test.csv
# $2 predict file (output file)

wget https://github.com/Liu-Shi-Tang/ML_model/releases/download/hw3/both_mean.npy
wget https://github.com/Liu-Shi-Tang/ML_model/releases/download/hw3/both_std.npy
wget https://github.com/Liu-Shi-Tang/ML_model/releases/download/hw3/mcp-best-acc-0.68250.h5
wget https://github.com/Liu-Shi-Tang/ML_model/releases/download/hw3/mcp-both-acc-0.68450.h5
python test.py $1 $2

