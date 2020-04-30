## a very si-mple example
import tax2vec as t2v
from tax2vec.preprocessing import *
from tax2vec.models import *
import numpy as np

def example_run(semantic_features=10, heuristic = "closeness_centrality"):
    labels, d_corpus,class_names = generate_corpus("../datasets/PAN_2016_age_srna_en.csv.gz",100000000000) ## the number is max number of docs per user

    ## use this tokenizer on whole corpus to guarantee equal splits! Note that train tokenizer is used again to not include any test info!
    sequence_word_matrix, _, _ = data_docs_to_matrix(d_corpus, mode="index_word")
    split_gen = split_generator(sequence_word_matrix, d_corpus, labels, num_splits=1, test=0.1)

    ## do the stratified shufflesplit
    for traintest in split_gen:
        (train_x,test_x,train_y,test_y) = traintest

        train_sequences,tokenizer,mlen = data_docs_to_matrix(train_x, mode="index_word")
        test_sequences = tokenizer.texts_to_sequences(test_x)    

        train_matrices_for_svm = []
        test_matrices_for_svm = []
        
        ## get the word index mappings for hypernym mappings
        dmap = tokenizer.__dict__['word_index']
        if semantic_features !=0:
            tax2vec_instance = t2v.tax2vec(max_features=semantic_features,
                                           targets=train_y,
                                           num_cpu=8,
                                           heuristic=heuristic,
                                           class_names=class_names)
            semantic_features_train = tax2vec_instance.fit_transform(train_sequences, dmap)

            ## get test features -- t2v
            semantic_features_test = tax2vec_instance.transform(test_sequences)
            train_matrices_for_svm.append(semantic_features_train)
            test_matrices_for_svm.append(semantic_features_test)

        ## word-level features
        tfidf_word_train, tokenizer_2, _ = data_docs_to_matrix(train_x, mode="matrix_word",max_features=10000)
        tfidf_word_test = tokenizer_2.transform(test_x)
        train_matrices_for_svm.append(tfidf_word_train)
        test_matrices_for_svm.append(tfidf_word_test)

        ## char-level features
        tfidf_char_train, tokenizer_3, _ = data_docs_to_matrix(train_x, mode="matrix_char",max_features=60)
        tfidf_char_test = tokenizer_3.transform(test_x)
        train_matrices_for_svm.append(tfidf_char_train)
        test_matrices_for_svm.append(tfidf_char_test)

        ## stack features (sparse)
        features_train = hstack(train_matrices_for_svm)
        features_test = hstack(test_matrices_for_svm)

        ## run the SVM, where C=50
        tmp_result = linear_SVM(features_train,features_test,train_y,test_y,cparam=50)

        return tmp_result

if __name__ == "__main__":

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    output = []
    for heuristic in ["closeness_centrality","rarest_terms","mutual_info","pagerank"]:
        for fn in [0,5,10,15,20,25,50,100,200,500,1000]:
            score = example_run(fn, heuristic = heuristic)
            output.append([fn,score, heuristic])

    dfx = pd.DataFrame(output)
    dfx.columns = ['Number of features','Performance (F1)','Heuristic']
    sns.lineplot(dfx['Number of features'], dfx['Performance (F1)'], hue = dfx['Heuristic'])
    plt.tight_layout()
    plt.savefig("../benchmark.png",dpi = 300)
        
    # print("No semantic features")
    # example_run(0)
    # print("10 semantic features")
    # example_run(10)
    # print("50 semantic features")
    # example_run(50)
