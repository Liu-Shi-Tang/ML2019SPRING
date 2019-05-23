

# $1 : image path
# $2 : test_case.csv path 
# $3 : predict file name(containing path)

wget https://github.com/Liu-Shi-Tang/ML_model/releases/download/HW7/best.h5
wget https://github.com/Liu-Shi-Tang/ML_model/releases/download/HW7/encoder.h5


python testProb.py ${1} ${2} ${3}
