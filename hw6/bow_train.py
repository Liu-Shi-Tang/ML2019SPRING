import jieba  # For cutting word
import sys

train_x_file_name = sys.argv[1]
train_y_file_name = sys.argv[2]
path_dict         = sys.argv[3]

# Load dict from TA
jieba.dt.tmp_dir = "./" # To set ./ as cache dir
jieba.load_userdict(path_dict)


train_x = []
train_y = []

# read train x
with open(train_x_file_name,'r',encoding = 'utf-8') as f :
    lines = f.readlines()
    for i,line in enumerate(lines) :
        # because there is repeated training datat after 119017
        if i > 119018 :
            break ;

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
        # because there is repeated training datat after 119017
        if i > 119018 :
            break ;

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

# Load pretrain Word2Vec model
model = Word2Vec.load("word2vec.model")

###########################################start to using keras######################################

import numpy as np
np.random.seed(9527)
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, GRU, Dense, LSTM, Dropout
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger


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
    trainable=True)

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

train_x_wv_bow = np.zeros((len(train_x_wv),len(word2idx)+1),dtype=np.int8)
for i,sentence in enumerate( train_x_wv) :
    print(i)
    for w in sentence :
        try :
            train_x_wv_bow[i,w] += 1 
        except :
            pass
print("success")
print(len(word2idx))


# test code
print (train_x_wv.shape)
print("train_x_wv[0]" + str(train_x_wv[0]))


# For one-hot encoding
train_y_one_hot = to_categorical(train_y,2)

# test code
print(train_y_one_hot.shape)
print("train_y_one_hot[0]:" , str(train_y_one_hot[0]))

def getModel3(in_shape) :
    model3 = Sequential()
    model3.add(Dense(1024,input_shape=in_shape,activation='sigmoid'))
    model3.add(Dense(1024,activation='sigmoid'))
    model3.add(Dense(512,activation='sigmoid'))
    model3.add(Dense(512,activation='sigmoid'))
    model3.add(Dense(2,activation='softmax'))
    model3.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model3

myRNNModel3 = getModel3(train_x_wv_bow[0].shape)
myRNNModel3.summary()



#history = myRNNModel.fit(x=train_x_wv,y=train_y_one_hot,batch_size=128,epochs=9,validation_split=0.1)
csv_logger3 = CSVLogger('log3.csv', append=False)
learning_rate3 = ReduceLROnPlateau(monitor='val_acc', patience=1, verbose=1, min_delta=1e-4, min_lr=1e-6)
checkpoint3 = ModelCheckpoint(filepath='best3.h5', monitor='val_acc', verbose=1, save_best_only=True)
early_stop3 = EarlyStopping(monitor='val_acc', patience=2, verbose=1)

history3 = myRNNModel3.fit(x=train_x_wv_bow,y=train_y_one_hot,batch_size=256,epochs=30,validation_split=0.1,shuffle=True,callbacks=[learning_rate3, checkpoint3, early_stop3, csv_logger3])



p5_1 = '在說別人白痴之前，先想想自己'
p5_2 = '在說別人之前先想想自己，白痴'

p5_1_list = []
p5_2_list = []

for w in jieba.cut(p5_1,cut_all=False) :
  p5_1_list.append(w)

for w in jieba.cut(p5_2,cut_all=False) :
  p5_2_list.append(w)

p5_list = []
p5_list.append(p5_1_list)
p5_list.append(p5_2_list)


p5_list = text_to_index(p5_list)

p5_list_bow = np.zeros((len(p5_list),len(word2idx)+1),dtype=np.int8)
for i,sentence in enumerate( p5_list) :
    print(i)
    for w in sentence :
        try :
            p5_list_bow[i,w] += 1 
        except :
            pass
print("success")
print(len(p5_list_bow))



print(myRNNModel3.predict(p5_list_bow))
