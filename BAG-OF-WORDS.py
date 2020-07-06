# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 12:14:14 2020

@author: micha
"""

import tensorflow.compat.v1 as tf
import numpy as np
emb_dims = 128
learning_rate = 0.001
tf.disable_eager_execution()
#-------------------------------------------------
# to one hot the words
#------------------------------------------------
def one_hot(ind,vocab_size):
    rec = np.zeros(vocab_size)
    rec[ind] = 1
    return rec


#----------------------------------------------------
# Create training data
#----------------------------------------------------
def create_training_data(corpus_raw,WINDOW_SIZE = 2):
    words_list = []

    for sent in corpus_raw.split('.'):
        for w in sent.split():
            if w != '.':  
                words_list.append(w.split('.')[0])       # Remove if delimiter is tied to the end of a word

    words_list = set(words_list)                     # Remove the duplicates for each word 

    word2ind = {}                                    # Define the dictionary for converting a word to index
    ind2word = {}                                    # Define dictionary for retrieving a word from its index

    vocab_size = len(words_list)                      # Count of unique words in the vocabulary

    for i,w in enumerate(words_list):                 # Build the dictionaries  
        word2ind[w] = i
        ind2word[i] = w
        
    print(word2ind)
    sentences_list = corpus_raw.split('.')
    sentences = []

    for sent in sentences_list:
        sent_array = sent.split()
        sent_array = [s.split('.')[0] for s in sent_array]
        sentences.append(sent_array)               # finally sentences would hold arrays of word array for sentences
    
    data_recs = []                                   # Holder for the input output record

    

    for sent in sentences:
        for ind,w in enumerate(sent):
            rec = []
            for nb_w in sent[max(ind - WINDOW_SIZE, 0) : min(ind + WINDOW_SIZE, len(sent)) + 1] : 
                if nb_w != w:
                    rec.append(nb_w)
                data_recs.append([rec,w])
    
    x_train,y_train = [],[]

    for rec in data_recs:
        input_ = np.zeros(vocab_size)
        for i in range(WINDOW_SIZE-1):
            input_ += one_hot(word2ind[ rec[0][i] ], vocab_size)
        input_ = input_/len(rec[0])
        x_train.append(input_)
        y_train.append(one_hot(word2ind[ rec[1] ], vocab_size))
        
    return x_train,y_train,word2ind,ind2word,vocab_size

corpus_raw = "Una vera e propria messa in stato d’accusa dei suoi stessi accusatori. Palamara, infatti, riconosce di aver fatto parte del sistema delle correnti, quel sistema che ora mi condanna, spesso mi insulta, perché a torto o a ragione individua in me l’unico responsabile di tutto. Io – dice l’ex presidente dell’Anm – non mi sottrarrò alle responsabilità politiche del mio operato per aver accettato regole del gioco sempre più discutibili. Ma deve essere chiaro che non ho mai agito da solo. Sarebbe troppo facile pensare questo. Il magistrato al centro dell’inchiesta che imbarazza tutto il mondo delle toghe, insomma, rivendica un passaggio fondamentale: non è solo lui l’artefice della degenerazione rappresentata dal sistema delle correnti. All’inizio – sostiene il pm – ero animato dal sacro fuoco del cambiamento, perché ovviamente anche io mi rendevo conto che era un meccanismo infernale, dal quale però mi sono lasciato inghiottire. Ma ciò non per sete di potere, bensì in una logica – che oggi riconosco, comunque, erronea – secondo cui il rafforzamento della posizione, mia e del mio gruppo di appartenenza, avrebbe potuto assicurare opportunità di avanzamento di colleghi meritevoli. Ma il fine, ora non posso non ammetterlo, non giustifica mai i mezzi."

corpus_raw = (corpus_raw).lower()
#----------------------------------------------------------------------
# Invoke the training data generation the corpus data
#-----------------------------------------------------------------------
x_train,y_train,word2ind,ind2word,vocab_size= create_training_data(corpus_raw,2)

#---------------------------------------------
# Build the Neural Net and Invoke training
#---------------------------------------------
# Placeholders for Input output
#----------------------------------------------
x = tf.placeholder(tf.float32,[None,vocab_size])
y = tf.placeholder(tf.float32,[None,vocab_size])
#---------------------------------------------
# Define the Embedding matrix weights and a bias
#----------------------------------------------
W = tf.Variable(tf.random_normal([vocab_size,emb_dims],mean=0.0,stddev=0.02,dtype=tf.float32))
b = tf.Variable(tf.random_normal([emb_dims],mean=0.0,stddev=0.02,dtype=tf.float32))
W_outer = tf.Variable(tf.random_normal([emb_dims,vocab_size],mean=0.0,stddev=0.02,dtype=tf.float32))
b_outer = tf.Variable(tf.random_normal([vocab_size],mean=0.0,stddev=0.02,dtype=tf.float32))

hidden = tf.add(tf.matmul(x,W),b)
logits = tf.add(tf.matmul(hidden,W_outer),b_outer)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

epochs,batch_size = 100,10
batch = len(x_train)//batch_size

# train for n_iter iterations
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        batch_index = 0 
        for batch_num in range(batch):
            x_batch = x_train[batch_index: batch_index +batch_size]
            y_batch = y_train[batch_index: batch_index +batch_size]
            sess.run(optimizer,feed_dict={x: x_batch,y: y_batch})
            print('epoch:',epoch,'loss :', sess.run(cost,feed_dict={x: x_batch,y: y_batch}))
    W_embed_trained = sess.run(W)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
W_embedded = TSNE(n_components=2).fit_transform(W_embed_trained)
plt.figure(figsize=(10,10))
for i in range(len(W_embedded)):
    plt.text(W_embedded[i,0],W_embedded[i,1],ind2word[i])

plt.xlim(-150,150)
plt.ylim(-150,150)