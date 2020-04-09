import tax2vec as t2v
from tax2vec.preprocessing import *
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

## For some data sets, semantic features do not help!

insults_dataset_train = pd.read_csv("../datasets/insults/train.tsv", sep = "\t")
insults_dataset_dev = pd.read_csv("../datasets/insults/dev.tsv", sep = "\t")
insults_dataset_test = pd.read_csv("../datasets/insults/test.tsv", sep = "\t")

train_text = insults_dataset_train['text_a'].values.tolist()
dev_text = insults_dataset_dev['text_a'].values.tolist()

train_targets = insults_dataset_train['label'].values.tolist()
dev_targets = insults_dataset_dev['label'].values.tolist()

## we are doing gridsearchCV, so this shall be merged.
train_text = train_text + dev_text
train_targets = train_targets + dev_targets

test_text = insults_dataset_test['text_a'].values
test_targets = insults_dataset_test['label'].values

final_output = []
for depth in [1,2,3,4,5]:
    for heuristic in ["closeness_centrality","rarest_terms","mutual_info","pagerank"]:
        for sem_feat_num in [0,10,100,1000,3000,15000]:
            ## trainset part
            train_sequences, tokenizer, mlen = data_docs_to_matrix(train_text, mode="index_word",simple_clean=True) ## simple clean removes english stopwords -> this is very basic preprocessing.

            dmap = tokenizer.__dict__['word_index']

            train_matrices = []
            test_matrices = []

            if sem_feat_num > 0:

                ## optionally feed targets=target_matrix for supervised feature construction
                ## start_term_depth denotes how high in the taxonomy must a given feature be to be considered
                tax2vec_instance = t2v.tax2vec(max_features=sem_feat_num, num_cpu=8, heuristic=heuristic, disambiguation_window = 3, start_term_depth = depth, targets = np.array(train_targets))

                semantic_features_train = tax2vec_instance.fit_transform(train_sequences, dmap)
                train_matrices.append(semantic_features_train)

                ## to obtain test features, simply transform.
                test_sequences = tokenizer.texts_to_sequences(test_text) ## tokenizer is already fit on train data

                test_matrices.append(tax2vec_instance.transform(test_sequences))

            tfidf_word_train, tokenizer_2, _ = data_docs_to_matrix(train_text, mode="matrix_word",max_features=10000)
            
            tfidf_word_test = tokenizer_2.transform(test_text)
            
            train_matrices.append(tfidf_word_train)
            test_matrices.append(tfidf_word_test)

            final_train = hstack(train_matrices)
            final_test = hstack(test_matrices)
            
            ## do the learning.
            lr = LogisticRegression(max_iter = 100000)
            parameters = {"C":[1,10,100,1000],"class_weight":["balanced",None]}
            clf = GridSearchCV(lr, parameters,verbose = 1).fit(final_train, train_targets)

            predictions = clf.predict(final_test)

            print("Depth: {}, Heuristic: {}, t2v features: {}, final accuracy: {}".format(depth, heuristic, sem_feat_num, accuracy_score(predictions,test_targets)))

            final_output.append((depth, heuristic, sem_feat_num, accuracy_score(predictions,test_targets)))

output_dfx = pd.DataFrame(final_output)
output_dfx.columns = ["depth","heuristic","semantic features","accuracy"]
print(output_dfx)
output_dfx.to_csv("results.tsv",sep = "\t")
