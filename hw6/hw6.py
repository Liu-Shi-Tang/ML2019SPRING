import jieba  # For cutting word
import sys

# Load dict from TA
path_dict = "dict.txt.big/dict.txt.big"
jieba.dt.tmp_dir = "./"
jieba.load_userdict(path_dict)

#####################################Using jieba#############################################


# # For test
# sequence = "我來到一個島...阿!還有early simple base line"
# 
# # cut a sequence by full mode (and get a generator)
# setList = jieba.cut(sequence,cut_all=True)
# 
# # extract word from generator (which is return by jieba.cut)
# wordList = []
# for c in setList :
#     wordList.append(c)
# 
# # print it out
# print("print it out:")
# for w in wordList :
#     print (w)

############################################################################################

train_x_file_name = sys.argv[1]
train_y_file_name = sys.argv[2]

train_x = []
train_y = []

# read train x
with open(train_x_file_name,'r',encoding = 'utf-8') as f :
    lines = f.readlines()
    for i,line in enumerate(lines) :
        # ignore first line "id,comment"
        if i != 0 :
            # words = line.split(',',1)
            # word = words[1]
            # train_x.append(word)
            train_x.append(line.split(',',1)[1])

# read train label
with open(train_y_file_name,'r',encoding = 'utf-8') as f :
    lines = f.readlines()
    for i,line in enumerate(lines) :
        # ignore first line "id,comment"
        if i != 0 :
            train_y.append(line.split(',',1)[1])

# cut train x 
cutWords = []
for x in train_x :
    # Using accurate mode
    setList = jieba.cut(x,cut_all=False)
    cutWords.append([])
    for w in setList :
        cutWords[-1].append(w)

# observe the cut words of 10-th line
print(cutWords[10])


############################################training word2Vec#########################################
from gensim.models import Word2Vec

# reference : https://radimrehurek.com/gensim/models/word2vec.html
model = Word2Vec(cutWords,size=100,window=5,min_count=1,workers=4)
model.save("word2vec.model")

















