## how to use tax2vec for xploratory tasks?

## a very si-mple example
import tax2vec as t2v
import pandas as pd
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

        ## inititalize tax2vec object adn fit and transform training data
        tax2vec_instance = t2v.tax2vec(max_features=50,
                                       num_cpu=8,
                                       heuristic="pagerank",
                                       mode="index_word")
        tax2vec_instance.fit_transform(train_x)

        ## print rankings
        for a, b in zip(tax2vec_instance.semantic_candidates,
                        tax2vec_instance.pagerank_scores):
            print("{} with score: {}".format(a, b))

        ## use tax2vec.WN to access hypernym graph


if __name__ == '__main__':
    run()
