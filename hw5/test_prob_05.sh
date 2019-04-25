


for i in {000..199};
do
  echo -e "diff ./attacked_images/$i.png ./test/$i.png"
  diff ./attacked_images/$i.png ./test/$i.png
done

