
# $1 : train.csv
wget https://github.com/Liu-Shi-Tang/ML_model/releases/download/hw3/mcp-best-acc-0.68250.h5
python generate_confusion_data.py $1
python confusion_matrix.py


