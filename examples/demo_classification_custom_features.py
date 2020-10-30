## more sophisticated features for classification -- this wins on most datasets!

## a very si-mple example
import tax2vec as t2v
from tax2vec.preprocessing import *
from tax2vec.models import *
from sklearn.model_selection import train_test_split


def run():
    PAN_dataset = pd.read_csv("../datasets/PAN_2016_age_srna_en.tsv", sep="\t")

    ## generate learning examples
    num_splits = 1

    ## do the stratified shufflesplit
    for _ in range(num_splits):

        (train_x, test_x, train_y,
         test_y) = train_test_split(PAN_dataset['text'].values.tolist(),
                                    PAN_dataset['class'].values.tolist(),
                                    test_size=0.1)

        tax2vec_instance = t2v.tax2vec(
            max_features=50,
            targets=train_y,
            num_cpu=8,
            heuristic="closeness_centrality",
            class_names=PAN_dataset['age_group'].values.tolist())
        semantic_features_train = tax2vec_instance.fit_transform(train_x)

        ## get test features
        train_matrices_for_svm = []
        test_matrices_for_svm = []
        semantic_features_test = tax2vec_instance.transform(test_x)

        train_matrices_for_svm.append(semantic_features_train)
        test_matrices_for_svm.append(semantic_features_test)

        tfidf_word_train, tokenizer_2, _ = data_docs_to_matrix(
            train_x, mode="matrix_pan")
        tfidf_word_test = tokenizer_2.transform(build_dataframe(test_x))
        train_matrices_for_svm.append(tfidf_word_train)
        test_matrices_for_svm.append(tfidf_word_test)

        ## stack features (sparse)
        features_train = hstack(train_matrices_for_svm)
        features_test = hstack(test_matrices_for_svm)

        ## run the SVM, where C=50
        tmp_result = linear_SVM(features_train,
                                features_test,
                                train_y,
                                test_y,
                                cparam=50)


if __name__ == '__main__':
    run()
