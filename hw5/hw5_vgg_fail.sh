
# $1 intput image dir
# $2 output image dir

for i in {0..180..20}
do
  python attack.py $i $1 $2 ;
done
