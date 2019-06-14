## some simple models

from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
import numpy as np
try:
    import tensorflow as tf
    from tensorflow import keras
except:
    pass

try:
    from tensorflow.python.keras.layers.advanced_activations import ELU
except:
    ELU = tf.keras.backend.elu

#from SOTAs.models.attn_lstm_hierarchical import *
    
def linear_SVM(train_x,test_x,train_y,test_y,cparam=1):

    new_train_y = []
    new_test_y = []
    
    for y in train_y:
        new_train_y.append(list(y).index(1))

    for y in test_y:
        new_test_y.append(list(y).index(1))
         
    clf = svm.LinearSVC(C=cparam)        
    clf.fit(train_x, new_train_y)
    
    y_pred = clf.predict(test_x)
    copt = f1_score(new_test_y, y_pred, average='micro')
            
    print("Current score {}".format(copt))
    return copt

def nn_batch_generator(X_data, y_data, batch_size):
    samples_per_epoch = X_data.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    index = np.arange(np.shape(y_data)[0])
    while 1:
        index_batch = index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X_data[index_batch,:].todense()
        y_batch = y_data[index_batch]
        counter += 1
        yield np.array(X_batch),y_batch
        if (counter > number_of_batches):
            counter=0

def simple_dnn(train_x,test_x,train_y,test_y,dnn_setting = "1000,500,100",batch_size=10,epochs=10):

    train_x = train_x.tocsr()
    test_x = test_x.tocsr()
    model = keras.Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    layer_sizes = [int(x) for x in dnn_setting.split(",")]
    model.add(keras.layers.Dense(layer_sizes[0], activation='relu',input_shape=(train_x.shape[1],)))
    
    # Add another:
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(layer_sizes[1], activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(layer_sizes[2], activation='relu'))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    
    # Add a softmax layer with 10 output units:
    model.add(keras.layers.Dense(train_y.shape[1], activation='softmax'))

    model.compile(optimizer=tf.train.AdamOptimizer(0.0001),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

    batchgen = nn_batch_generator(train_x,train_y,batch_size=batch_size)

    for j in range(epochs):
        model.fit_generator(generator=batchgen,
                            use_multiprocessing=False,
                            workers=1,verbose=1,steps_per_epoch=1)

    mask = np.zeros((test_y.shape[0],test_y.shape[1]))
    preds = model.predict(test_x)
    mx = np.argmax(preds, axis=1)
    for j in np.arange(mask.shape[0]):        
        mask[j, mx[j]] = 1
        
    copt = f1_score(test_y, mask, average='micro')
    print("Current score {}".format(copt))
    return copt

def SRNA(train_features,test_features,train_y,test_y,maxlen=0,epoch_num=5):

        ## process data
        x_train,semantic_train = train_features
        x_test,semantic_test = test_features
        labels_train = train_y
        labels_test = test_y
        
        # set parameters
        max_features = 120000
        semantic_embedding_dims = 100
        batch_size = 48
        embedding_dims = 300#int(maxlen/2)
        filters = embedding_dims
        kernel_size = 5
        hidden_dims = filters
        epochs = epoch_num
        
        semantic_shape_hidden = hidden_dims
        semantic_shape = semantic_train.shape[1]
        if maxlen > 0:
            x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
            x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

        ## a hybrid model with two inputs!
        input1 = tf.keras.layers.Input(shape=(x_train.shape[1],))
        e1 = tf.keras.layers.Embedding(max_features,embedding_dims)(input1)
        d1 = tf.keras.layers.Dropout(0.5)(e1)
#        c1 = tf.keras.layers.Conv1D(filters,
#                    kernel_size,
#                    padding='valid',
#                    activation='relu',
#                    strides=1)(d1)
        c0 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(300, return_sequences=False, dropout=0.25,recurrent_dropout=0.25,activation='softmax'))(d1)
#        gp1 = tf.keras.layers.GlobalMaxPooling1D()(c1)
        de1 = tf.keras.layers.Dense(hidden_dims)(c0)
        d1_1 = tf.keras.layers.Dropout(0.3)(de1)
        
        input2 = tf.keras.layers.Input(shape=(semantic_shape,))
        #activation_1 = ELU()(e2_2)
        d2_1 = tf.keras.layers.Dense(semantic_shape_hidden)(input2)
        drop_2 = tf.keras.layers.Dropout(0.2)(d2_1)
        pm = ELU()(drop_2)
        
        added = tf.keras.layers.Add()([d1_1, pm])
        mix1 = tf.keras.layers.Dense(100)(added)
        dp_1 = tf.keras.layers.Dropout(0.3)(mix1)
        da_2 = ELU()(dp_1)
        mix2 = tf.keras.layers.Dense(50)(da_2)
        out = tf.keras.layers.Dense(labels_train.shape[1],activation="sigmoid")(mix2)
        model = tf.keras.models.Model(inputs=[input1, input2], outputs=out)

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['categorical_accuracy'])
        callbacks = [
          #  EarlyStoppingByLossVal(monitor='loss', value=0.2, verbose=1)
        ]
        
        print(model.summary())
        
        model.fit([x_train,semantic_train], labels_train,
                       batch_size=batch_size,
                       epochs=epochs)
        
#        best_model = return_best_hybrid_model(model,[x_train,semantic_train],labels_train,epochs=20)

        predictions = model.predict([x_test,semantic_test])
        labels = np.zeros(predictions.shape)
        labels[predictions>0.5] = 1
        copt = f1_score(test_y, labels, average='micro')
        print("Current result {}".format(copt))
        return copt

# def hierarchical_attention(train_features,test_features,train_y,test_y,tokenizer,parameter_dict=None):

#     print(tokenizer.__dict__.keys())
#     vocab_size = len(tokenizer.__dict__['word_index']) + 1
    
#     ## make validation dataset first
#     print(parameter_dict)
#     predictions = attention_hilstm(train_features,test_features,train_y,test_y,vocab_size,parameter_dict)
#     print("Current opt score: {}".format(predictions))
#     return predictions

# def linear_tf(train_features,test_features,train_y,test_y):

#     ## train a single linear classifier!
#     ## yield f1score..
#     m1 = tf.convert_to_tensor(train_features, dtype=tf.float32)
#     m2 = tf.convert_to_tensor(train_y, dtype=tf.float32)

#     def input_fn_train():
#         return (m1,m2)
    
#     estimator = tf.estimator.LinearClassifier()
#     estimator.train(input_fn=input_fn_train)
        
#     pass
