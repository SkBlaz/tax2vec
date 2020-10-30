## a very si-mple example
import tax2vec as t2v
from tax2vec.preprocessing import *
from tax2vec.models import *
from sklearn.model_selection import train_test_split


def example_run(semantic_features=10,
                heuristic="closeness_centrality",
                num_splits=1):

    PAN_dataset = pd.read_csv("../datasets/PAN_2016_age_srna_en.tsv", sep="\t")

    ## do the stratified shufflesplit
    for _ in range(num_splits):

        (train_x, test_x, train_y,
         test_y) = train_test_split(PAN_dataset['text'].values.tolist(),
                                    PAN_dataset['class'].values.tolist(),
                                    test_size=0.1)

        train_matrices_for_svm = []
        test_matrices_for_svm = []

        ## get the word index mappings for hypernym mappings
        if semantic_features != 0:
            tax2vec_instance = t2v.tax2vec(
                max_features=semantic_features,
                targets=train_y,
                num_cpu=8,
                heuristic=heuristic,
                class_names=PAN_dataset['age_group'].values.tolist())
            semantic_features_train = tax2vec_instance.fit_transform(train_x)

            ## get test features -- t2v
            semantic_features_test = tax2vec_instance.transform(test_x)
            train_matrices_for_svm.append(semantic_features_train)
            test_matrices_for_svm.append(semantic_features_test)

        ## word-level features
        tfidf_word_train, tokenizer_2, _ = data_docs_to_matrix(
            train_x, mode="matrix_word", max_features=10000)
        tfidf_word_test = tokenizer_2.transform(test_x)
        train_matrices_for_svm.append(tfidf_word_train)
        test_matrices_for_svm.append(tfidf_word_test)

        ## char-level features
        tfidf_char_train, tokenizer_3, _ = data_docs_to_matrix(
            train_x, mode="matrix_char", max_features=60)
        tfidf_char_test = tokenizer_3.transform(test_x)
        train_matrices_for_svm.append(tfidf_char_train)
        test_matrices_for_svm.append(tfidf_char_test)

        ## stack features (sparse)
        features_train = hstack(train_matrices_for_svm)
        features_test = hstack(test_matrices_for_svm)

        ## run the SVM, where C=50
        tmp_result = linear_SVM(features_train,
                                features_test,
                                train_y,
                                test_y,
                                cparam=50)

        return tmp_result


if __name__ == "__main__":

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import time
    start = time.time()
    output = []
    for heuristic in [
            "betweenness_centrality", "closeness_centrality", "rarest_terms",
            "mutual_info", "pagerank"
    ]:
        for fn in [0, 5, 10, 15, 20, 25, 50, 100, 200, 500, 1000]:
            score = example_run(fn, heuristic=heuristic)
            output.append([fn, score, heuristic])

    dfx = pd.DataFrame(output)
    dfx.columns = ['Number of features', 'Performance (F1)', 'Heuristic']
    sns.lineplot(dfx['Number of features'],
                 dfx['Performance (F1)'],
                 hue=dfx['Heuristic'])
    plt.tight_layout()
    plt.savefig("../benchmark.png", dpi=300)

    # print("No semantic features")
    # example_run(0)
    # print("10 semantic features")
    # example_run(10)
    # print("50 semantic features")
    # example_run(50)
