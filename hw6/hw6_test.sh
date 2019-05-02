

# $1 : test_x file
# $2 : dict.txt.big file
# $3 : result  file

# default path
testX="392045_data/test_x.csv"
dictTxtBig="./dict.txt.big/dict.txt.big"
result="resul.csv"

if [ $# -eq 3 ]
then
  echo "get 3 arg"
  testX=$1
  dictTxtBig=$2
  result=$3
fi

# testing
python test.py ${testX} ${dictTxtBig} ${result}



