## how to use tax2vec for xploratory tasks?

## a very si-mple example
from tax2vec.preprocessing import *
import tax2vec as t2v
import numpy as np

labels, d_corpus,class_names = generate_corpus("../datasets/PAN_2016_age_srna_en.csv.gz",100000000000) ## the number is max number of docs per user

## use this tokenizer on whole corpus to guarantee equal splits! Note that train tokenizer is used again to not include any test info!
sequence_word_matrix, _, _ = data_docs_to_matrix(d_corpus, mode="index_word")
split_gen = split_generator(sequence_word_matrix, d_corpus, labels, num_splits=1, test=0.1)

## generate learning examples
result_vector = []

## do the stratified shufflesplit
for traintest in split_gen:
    (train_x,test_x,train_y,test_y) = traintest
    train_sequences,tokenizer,mlen = data_docs_to_matrix(train_x, mode="index_word")
    test_sequences = tokenizer.texts_to_sequences(test_x)

    ## get the word index mappings for hypernym mappings
    dmap = tokenizer.__dict__['word_index']
    tax2vec_instance = t2v.tax2vec(max_features=50, document_split_symbol="|||",targets=train_y,num_cpu=8,heuristic="mutual_info",class_names=class_names)
    semantic_features_train = tax2vec_instance.fit_transform(train_sequences, dmap)
    
    print(tax2vec_instance.semantic_candidates)
    print(tax2vec_instance.top_mutual_information_scores)
    for x,y,z in zip(tax2vec_instance.semantic_candidates,tax2vec_instance.top_mutual_information_scores,tax2vec_instance.relevant_classes):
        topic_string = " ".join([str(x[0])+":"+str(np.around(x[1],2)) for x in z])
        print(x,np.round(y,3),topic_string)
        
    ## use tax2vec.WN to access hypernym graph

    
