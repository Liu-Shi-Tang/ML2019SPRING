

# $1 : train_x file
# $2 : train_y file
# $3 : test_x  file
# $4 : dict.txt.big file

# default path
trainX="./392045_data/train_x.csv"
trainY="./392045_data/train_y.csv"
testX="392045_data/test_x.csv"
dictTxtBig="./dict.txt.big/dict.txt.big"

if [ $# -eq 4 ]
then
  echo "Get 4 arg"
  trainX=$1
  trainY=$2
  testX=$3
  dictTxtBig=$4
fi

# training gensim model
python trainW2v.py ${trainX} ${trainY} ${testX} ${dictTxtBig}

# training keras model
python trainKeras.py ${trainX} ${trainY} ${dictTxtBig}

