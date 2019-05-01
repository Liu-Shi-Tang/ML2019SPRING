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
# print(cutWords[10])


############################################training word2Vec#########################################
from gensim.models import Word2Vec

# reference : https://radimrehurek.com/gensim/models/word2vec.html
model = Word2Vec(cutWords,size=250,window=5,iter=10,min_count=1,workers=4,)
model.save("word2vec.model")

###########################################start to using keras######################################

import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, GRU, Dense



embedding_matrix = np.zeros((len(model.wv.vocab.items())+1 , model.vector_size))

word2idx = {}

vocab_list = []
for word , others in model.wv.vocab.items() :
    vocab_list.append((word , model.wv[word]))

for i , vocab in enumerate(vocab_list) :
    word , vector = vocab
    embedding_matrix[i+1] = vector
    word2idx[word] = i+1


embedding_layer = Embedding(
    input_dim=embedding_matrix.shape[0],
    output_dim=embedding_matrix.shape[1],
    weights=[embedding_matrix],
    trainable=False)

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



# input length
SEQUENCE_LENGTH = 50
train_x_wv = text_to_index(cutWords)
train_x_wv = pad_sequences(train_x_wv,maxlen = SEQUENCE_LENGTH)

# test code
print (train_x_wv.shape)
print("train_x_wv[0]" + str(train_x_wv[0]))


# For one-hot encoding
train_y_one_hot = to_categorical(train_y,2)

# test code
print(train_y_one_hot.shape)
print("train_y_one_hot[0]:" , str(train_y_one_hot[0]))


def getModel(emLayer) :
    model = Sequential()
    model.add(emLayer)
    model.add(GRU(32))
    model.add(Dense(300,activation='sigmoid'))
    model.add(Dense(200,activation='sigmoid'))
    model.add(Dense(100,activation='sigmoid'))
    model.add(Dense(2,activation='softmax'))
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model

myRNNModel = getModel(embedding_layer)
myRNNModel.summary()


history = myRNNModel.fit(x=train_x_wv,y=train_y_one_hot,batch_size=128,epochs=11,validation_split=0.1)


############################################### testing ###############################


test_x_file_name = sys.argv[3]


test_x = []

# read train x
with open(test_x_file_name,'r',encoding = 'utf-8') as f :
    lines = f.readlines()
    for i,line in enumerate(lines) :
        # ignore first line "id,comment"
        if i != 0 :
            # words = line.split(',',1)
            # word = words[1]
            # train_x.append(word)
            test_x.append(line.split(',',1)[1])


# cut train x 
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

result = myRNNModel.predict(test_x_wv)

print(result)
print(result.shape)
print(type(result))


with open('result.csv','w') as f :
    f.write('id,label\n')
    for i in range(len(result)) :
        f.write(str(i) + ',' + str(int(np.argmax(result[i]))) + '\n')
    f.close()
print("end")















