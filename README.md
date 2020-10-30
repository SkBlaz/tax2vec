# tax2vec

Semantic space vectorization algorithm, official repository of the paper:
https://www.sciencedirect.com/science/article/pii/S0885230820300371

## Short description

Tax2vec is a simple data enrichment approach. Its main goal is to extract corpus-relevant semantic information in the form of new features, directly useful for learning.
> Key idea: Inject semantic background knowledge into existing textual data.
![Key idea](workflow.png)

## Getting Started
Below you shall find instructions for installation of tax2vec library.

### Prerequisites

What things you need to install the software and how to install them follows next. Generic install:

```
pip install -r requirements.txt
```
And also,
```
conda install --yes --file requirements.txt
```

We've also prepared a conda env with all requirements, which can be initiated as:
```
conda env create -f environment.yml
```

### Installing

Installing is simple!

```
pip3 install tax2vec
```
And that's it. You can also install the library directly:

```
python3 setup.py install
```

Note that some of the tf-idf constructors use nltk addons. Tax2vec informs you when an addon is needed and how to install it. For example, to install punctuation, one does:

```
import nltk
nltk.download('punct')
```

## Tests
The minimal tests that need to pass in order to assure the library works OK can be run from the examples folder as:
```
bash run_all_tests.sh
```
## A self contained example

Assume you are given a blob of documents, and are asked to generate semantic features. The following example starts from a randomly selected Brexit Wikipedia article:

```python
## simplest possible use
import tax2vec as t2v
from tax2vec.preprocessing import *

def run():
    train_text = ["Brexit (/ˈbrɛksɪt, ˈbrɛɡzɪt/;[1] a portmanteau of British and exit) is the withdrawal of the United Kingdom (UK) from the European Union (EU). Following a referendum held on 23 June 2016 in which 51.9 per cent of those voting supported leaving the EU, the Government invoked Article 50 of the Treaty on European Union, starting a two-year process which was due to conclude with the UK's exit on 29 March 2019 – a deadline which has since been extended to 31 October 2019.[2]","Withdrawal from the EU has been advocated by both left-wing and right-wing Eurosceptics, while pro-Europeanists, who also span the political spectrum, have advocated continued membership and maintaining the customs union and single market. The UK joined the European Communities (EC) in 1973 under the Conservative government of Edward Heath, with continued membership endorsed by a referendum in 1975. In the 1970s and 1980s, withdrawal from the EC was advocated mainly by the political left, with the Labour Party's 1983 election manifesto advocating full withdrawal. From the 1990s, opposition to further European integration came mainly from the right, and divisions within the Conservative Party led to rebellion over the Maastricht Treaty in 1992. The growth of the UK Independence Party (UKIP) in the early 2010s and the influence of the cross-party People's Pledge campaign have been described as influential in bringing about a referendum. The Conservative Prime Minister, David Cameron, pledged during the campaign for the 2015 general election to hold a new referendum—a promise which he fulfilled in 2016 following pressure from the Eurosceptic wing of his party. Cameron, who had campaigned to remain, resigned after the result and was succeeded by Theresa May, his former Home Secretary. She called a snap general election less than a year later but lost her overall majority. Her minority government is supported in key votes by the Democratic Unionist Party.","The broad consensus among economists is that Brexit will likely reduce the UK's real per capita income in the medium term and long term, and that the Brexit referendum itself damaged the economy.[a] Studies on effects since the referendum show a reduction in GDP, trade and investment, as well as household losses from increased inflation. Brexit is likely to reduce immigration from European Economic Area (EEA) countries to the UK, and poses challenges for UK higher education and academic research. As of May 2019, the size of the divorce bill—the UK's inheritance of existing EU trade agreements—and relations with Ireland and other EU member states remains uncertain. The precise impact on the UK depends on whether the process will be a hard or soft Brexit."]

    test_text = ["When the European Communities (EC) came into being in 1958, the UK chose to remain aloof and instead join the alternative bloc, EFTA. Almost immediately the British government regretted its decision, and in 1961, along with Denmark, Ireland and Norway, the UK applied to join the three Communities. However, President Charles de Gaulle saw British membership as a Trojan horse for US influence, and vetoed it; all four applications were suspended. The four countries resubmitted their applications in 1967, and the French veto was lifted upon Georges Pompidou succeeding de Gaulle in 1969.[2] In 1970, accession negotiations took place between the UK Government, led by Conservative Prime Minister Edward Heath, the European Communities and various European leaders. Despite disagreements over the CAP and the UK's relationship with the Commonwealth, terms were agreed. In October 1971, after a lengthy Commons debate, MPs voted 356-244 in favour of joining the EEC."]

    ## optionally feed targets=target_matrix for supervised feature construction
    ## start_term_depth denotes how high in the taxonomy must a given feature be to be considered
    tax2vec_instance = t2v.tax2vec(max_features=10, num_cpu=2, heuristic="pagerank", disambiguation_window=2, start_term_depth=3) 

    print(train_text)
    ## to obtain train feature one must fit and transfrorm
    semantic_features_train = tax2vec_instance.fit_transform(train_text)
    print(semantic_features_train)
    ## to obtain test features, simply transform
    semantic_features_test = tax2vec_instance.transform(test_text)
    assert semantic_features_train.shape[1] == semantic_features_test.shape[1]

    ## which features are the most relevant?
    for a, b in zip(tax2vec_instance.semantic_candidates, tax2vec_instance.pagerank_scores):
       print("{} with score: {}".format(a, b))
       
if __name__ == '__main__':
    run()

```
yields:

| Hypernym            | Score  | PageRank             |
|---------------------|--------|----------------------|
| vote.v.05           | score: | 0.020820367993329862 |
| edward.n.01         | score: | 0.011273376619092617 |
| david.n.03          | score: | 0.011263940537596226 |
| portmanteau.n.02    | score: | 0.011254504456099833 |
| british.n.01        | score: | 0.011254504456099833 |
| passing.n.02        | score: | 0.011254504456099833 |
| withdrawal.n.01     | score: | 0.011254504456099833 |
| united_kingdom.n.01 | score: | 0.011254504456099833 |
| european.n.01       | score: | 0.011254504456099833 |
| union.n.11          | score: | 0.011254504456099833 |

## Basic use
>Key idea: Generate word-index mapping of the corpus, use this as input along with (optional) classes.

First, import the library and some of the preprocessing methods..

```python
import tax2vec as t2v
from tax2vec.preprocessing import *
```

Next, we load the corpus using the in-built methods. Note that any tokenizer can be used for this!

```python

PAN_dataset = pd.read_csv("../datasets/PAN_2016_age_srna_en.tsv", sep="\t")

# Get splits
(train_x,test_x,train_y,test_y) = train_test_split(PAN_dataset['text'].values.tolist(), PAN_dataset['class'].values.tolist(), test_size=0.1)

## tax2vec part
tax2vec_instance = t2v.tax2vec(max_features=30, targets=train_y, num_cpu=8, heuristic="closeness_centrality", class_names=PAN_dataset['age_group'].values.tolist())

## fit and transform
semantic_features_train = tax2vec_instance.fit_transform(train_x)

## just transform
semantic_features_test = tax2vec_instance.transform(test_x)

## And that's it!

```

# Making features from knowledge graphs?
(Beta feature) It is also possible to obtain features from e.g., Microsoft Concept Graph via function knowledge_graph_features() method. It uses SpaCy to find nouns in the sentence and then look up generalizations in the graph. To do this, we just need to pass argument *knowledge_graph=True* when e.g., using Microsoft Concept Graph. Also the path to the file with knowledge graph relations should be specified (parameter *path*). Parameter *hyp* specifies how many hypernyms from each relevant word from our text we should take in account. Example follows:

```python

PAN_dataset = pd.read_csv("../datasets/PAN_2016_age_srna_en.tsv", sep="\t")

(train_x,test_x,train_y,test_y) = train_test_split(PAN_dataset['text'].values.tolist(), PAN_dataset['class'].values.tolist(), test_size=0.1)

## tax2vec part - here are some extra parameters for knowledge graph, namely knowledge_graph, hyp and path
tax2vec_instance = t2v.tax2vec(max_features=10, num_cpu=8, heuristic="pagerank", disambiguation_window=2, start_term_depth=3, knowledge_graph=True, mode="index_word", simple_clean=True, hyp="all", path="../data-concept/refined.txt") 

## and fit and transform
semantic_features_train = tax2vec_instance.fit_transform(train_x)

## just transform
semantic_features_test = tax2vec_instance.transform(test_x)

```

The knowledge base is a text file with IsA relations in the form: instance *is_a* concept. The lines below are from file which was created from the Microsoft Concept Graph. So on the left side is the query word and on the right are the hypernyms.

```python
...
headache	is_a	symptom
twitter	is_a	social medium
diabetes	is_a	condition
stress	is_a	factor
aluminum	is_a	metal
basketball	is_a	sport
...

```

## Relevant hyperparameters
| Hyperparameter                   | Default value | Possible values                                                  |
|----------------------------------|---------------|------------------------------------------------------------------|
| max_features                     | 100           | int                                                              |
| disambiguation_window            | 3             | int                                                              |
| heuristic                        | "mutual_info" | ["closeness_centrality","rarest_terms","mutual_info","pagerank"] |
| num_cpu                          | "all"         | int or "all" - automatic detection                               |
| hypernym_distribution (optional) | somefile.npy  | "./hypernym_space/dist1.npy"                                     |
| targets                          | None          | numeric vector of targets (for supervised feature ranking)       |
| class_names                      | None          | names of classes                                                 |
| start_term_depth                 | 0             | terms at depth *larger than this* will be considered             |
| knowledge_graph		           | False	       | True, False						                              |
| hyp                              | "all"         | "all" or int                                                     |

## Common behavior
Current experiments indicate, that a rather small number of semantic features can greatly impact the classifier's performance. See examples to
reproduce the following benchmark, based on the PAN Age data set.

![Benchmark](benchmark.png)

## Example uses

To reproduce SOTA results on the classification task, you can run:
```
python3 demo_classification.py
```

To use custom feature constructor by Martinc et al. (2017)
```
python3 demo_classification_custom_features.py
```

To reproduce the explainability features:

```
python3 demo_explain_corpus.py
```

To use it in an unsupervised setting:

```
python3 demo_explain_unsupervised.py
```

And to create the semantic features via knowledge graph: (beta feature)

```
python3 demo_knowledge_graph.py
```

## Contributing

To contribute, simply open an issue or a pull request!

## Authors

tax2vec was created by Blaž Škrlj, Jan Kralj, Matej Martinc, Nada Lavrač and Senja Pollak.

## License

See LICENSE.md for more details.

## Citation
Please cite:

```
@article{SKRLJ2020101104,
title = "tax2vec: Constructing Interpretable Features from Taxonomies for Short Text Classification",
journal = "Computer Speech & Language",
pages = "101104",
year = "2020",
issn = "0885-2308",
doi = "https://doi.org/10.1016/j.csl.2020.101104",
url = "http://www.sciencedirect.com/science/article/pii/S0885230820300371",
author = "Blaž Škrlj and Matej Martinc and Jan Kralj and Nada Lavrač and Senja Pollak",
keywords = "taxonomies, vectorization, text classification, short documents, feature construction, semantic enrichment",
abstract = "The use of background knowledge is largely unexploited in text classification tasks. This paper explores word taxonomies as means for constructing new semantic features, which may improve the performance and robustness of the learned classifiers. We propose tax2vec, a parallel algorithm for constructing taxonomy-based features, and demonstrate its use on six short text classification problems: prediction of gender, personality type, age, news topics, drug side effects and drug effectiveness. The constructed semantic features, in combination with fast linear classifiers, tested against strong baselines such as hierarchical attention neural networks, achieves comparable classification results on short text documents. The algorithm’s performance is also tested in a few-shot learning setting, indicating that the inclusion of semantic features can improve the performance in data-scarce situations. The tax2vec capability to extract corpus-specific semantic keywords is also demonstrated. Finally, we investigate the semantic space of potential features, where we observe a similarity with the well known Zipf’s law."
}
```
