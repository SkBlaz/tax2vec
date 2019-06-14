## more sophisticated features for classification -- this wins on most datasets!

## a very si-mple example
import tax2vec as t2v
from tax2vec.preprocessing import *
from tax2vec.models import *
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
    tax2vec_instance = t2v.tax2vec(max_features=50, targets=train_y,num_cpu=8,heuristic="closeness_centrality",class_names=class_names)
    semantic_features_train = tax2vec_instance.fit_transform(train_sequences, dmap)

    ## get test features
    train_matrices_for_svm = []
    test_matrices_for_svm = []
    semantic_features_test = tax2vec_instance.transform(test_sequences)

    train_matrices_for_svm.append(semantic_features_train)
    test_matrices_for_svm.append(semantic_features_test)
    
    tfidf_word_train, tokenizer_2, _ = data_docs_to_matrix(train_x, mode="matrix_pan")
    tfidf_word_test = tokenizer_2.transform(build_dataframe(test_x))
    train_matrices_for_svm.append(tfidf_word_train)
    test_matrices_for_svm.append(tfidf_word_test)
    
    ## stack features (sparse)
    features_train = hstack(train_matrices_for_svm)
    features_test = hstack(test_matrices_for_svm)

    ## run the SVM, where C=50
    tmp_result = linear_SVM(features_train,features_test,train_y,test_y,cparam=50)

