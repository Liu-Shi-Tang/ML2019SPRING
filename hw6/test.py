import keras.models
from keras.models import load_model
import sys
import numpy as np
import jieba
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences

# Load pretrain Word2Vec model
model = Word2Vec.load("word2vec.model")


test_x_file_name = sys.argv[1]
path_dict        = sys.argv[2]
result_file_name = sys.argv[3]


# Load dict from TA
jieba.dt.tmp_dir = "./" # set cache dir
jieba.load_userdict(path_dict)


word2idx = {}

vocab_list = []
for word , others in model.wv.vocab.items() :
    vocab_list.append((word , model.wv[word]))

for i , vocab in enumerate(vocab_list) :
    word , vector = vocab
    word2idx[word] = i+1


def text_to_index(corpus) :
    new_corpus = []
    for doc in corpus :
        new_doc = []
        for word in doc :
            try :
                new_doc.append(word2idx[word])
            except :
                new_doc.append(0)
        new_corpus.append(new_doc)
    return np.array(new_corpus)



test_x = []

# read test x
with open(test_x_file_name,'r',encoding = 'utf-8') as f :
    lines = f.readlines()
    for i,line in enumerate(lines) :
        # ignore first line "id,comment"
        if i != 0 :
            # words = line.split(',',1)
            # word = words[1]
            # train_x.append(word)
            test_x.append(line.split(',',1)[1])


# cut test x 
test_cutWords = []
for x in test_x :
    # Using accurate mode
    setList = jieba.cut(x,cut_all=False)
    test_cutWords.append([])
    for w in setList :
        test_cutWords[-1].append(w)

# input length
SEQUENCE_LENGTH = 50
test_x_wv = text_to_index(test_cutWords)
test_x_wv = pad_sequences(test_x_wv,maxlen = SEQUENCE_LENGTH)

myRNNModel = load_model('best.h5')
result = myRNNModel.predict(test_x_wv)


with open(result_file_name,'w') as f :
    f.write('id,label\n')
    for i in range(len(result)) :
        f.write(str(i) + ',' + str(int(np.argmax(result[i]))) + '\n')
    f.close()


print("Produce {}".format(result_file_name))




