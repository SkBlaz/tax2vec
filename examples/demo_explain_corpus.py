## how to use tax2vec for xploratory tasks?

## a very si-mple example
from tax2vec.preprocessing import *
import tax2vec as t2v
import numpy as np
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

        ## get the word index mappings for hypernym mappings
        tax2vec_instance = t2v.tax2vec(
            max_features=50,
            document_split_symbol="|||",
            targets=train_y,
            num_cpu=8,
            heuristic="mutual_info",
            class_names=PAN_dataset['age_group'].values.tolist())
        tax2vec_instance.fit_transform(train_x)

        print(tax2vec_instance.semantic_candidates)
        print(tax2vec_instance.top_mutual_information_scores)
        for x, y, z in zip(tax2vec_instance.semantic_candidates,
                           tax2vec_instance.top_mutual_information_scores,
                           tax2vec_instance.relevant_classes):
            topic_string = " ".join(
                [str(x[0]) + ":" + str(np.around(x[1], 2)) for x in z])
            print(x, np.round(y, 3), topic_string)

        ## use tax2vec.WN to access hypernym graph


if __name__ == '__main__':
    run()
