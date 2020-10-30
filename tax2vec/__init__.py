##############################
# tax2vec --- Blaz Skrlj 2019
##############################

try:
    from tqdm import tqdm

    pbar = True
except BaseException:
    pbar = False
import multiprocessing
from collections import defaultdict
import numpy as np

try:
    import nltk

    nltk.data.path.append("./nltk_data")
except Exception as es:
    print(es)

from nltk.wsd import lesk
import multiprocessing as mp
from sklearn.feature_selection import mutual_info_classif
import networkx as nx
from collections import Counter
import operator
import scipy.sparse as sps
import logging
import spacy
import re
import tax2vec.preprocessing as prep

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)


class tax2vec:
    def __init__(self,
                 max_features=100,
                 disambiguation_window=3,
                 document_split_symbol="MERGERTAG",
                 heuristic="mutual_info",
                 num_cpu="all",
                 hypernym_distribution="./hypernym_space/dist1.npy",
                 targets=None,
                 class_names=None,
                 start_term_depth=0,
                 knowledge_graph=False,
                 mode="index_word",
                 simple_clean=False,
                 hyp='all',
                 path=None):
        '''
        Initiate the core properties
        '''

        self.start_term_depth = start_term_depth
        self.class_names = class_names
        self.targets = targets
        self.heuristic = heuristic
        self.hypernym_space = hypernym_distribution
        self.max_features = max_features
        self.disambiguation_window = disambiguation_window
        self.initial_transformation = False
        self.skip_transform = False
        self.indices_selected_features = None
        self.document_split_symbol = document_split_symbol
        self.initial_terms = []  # for personalized node ranking
        self.possible_heuristics = [
            "closeness_centrality", "rarest_terms", "mutual_info",
            "betweenness_centrality", "pagerank"
        ]
        self.reversed_wmap = None
        self.doc_seqs = None
        self.knowledge_graph = knowledge_graph
        if self.knowledge_graph:
            self.parallel = False
            self.nlp = spacy.load("en_core_web_sm")
        else:
            self.parallel = True
        self.mode = mode
        self.simple_clean = simple_clean
        self.hypernyms = hyp
        self.knowledge_graph_path = path

        if num_cpu == "all":
            self.num_cpu = mp.cpu_count()

        elif num_cpu == 1:
            self.parallel = False
        #            self.num_cpu = 1

        else:
            self.num_cpu = num_cpu
        self.monitor("Using {} heuristic.".format(self.heuristic))

    def monitor(self, message):
        '''
        A very simple state monitor
        '''

        logging.info(message)

    def fit(self, text):
        '''
        A simple fit method
        '''
        train_sequences, tokenizer, mlen = prep.data_docs_to_matrix(
            text, mode=self.mode, simple_clean=self.simple_clean
        )  # simple clean removes english stopwords -> this is very basic preprocessing.
        self.tokenizer = tokenizer
        dmap = tokenizer.__dict__['word_index']
        self.monitor("Constructing local taxonomy...")
        if self.knowledge_graph:
            self.knowledge_graph_features(train_sequences,
                                          dmap,
                                          hyp=self.hypernyms,
                                          path=self.knowledge_graph_path)
        else:
            self.wordnet_features(train_sequences, dmap)

    def disambiguation_synset(self, word, document, word_index=None):
        '''
        First split each multidocument and select most frequent hypernyms in terms of occurrence.
        '''

        if self.disambiguation_window is None:

            # take into account whole docs
            docs = " ".join(document).split(self.document_split_symbol)
            hyps = []
            for doc in docs:
                ww = " ".join(doc)
                hypernym = lesk(ww, word)
                if hypernym is not None:
                    hyps.append(hypernym)
            if len(hyps) > 0:
                Counter(hyps)
                return max(set(hyps), key=hyps.count)
            else:
                return None

        else:
            if (word_index - self.disambiguation_window) > 0:
                bottom_index = (word_index - self.disambiguation_window)
            else:
                bottom_index = 0

            if (word_index + self.disambiguation_window) > len(document):
                top_index = len(document)
            else:
                top_index = (word_index + self.disambiguation_window)

            word_window = document[bottom_index:top_index]
            ww = " ".join(word_window)
            hypernym = lesk(ww, word)

            if hypernym is not None:
                return hypernym
            else:
                return None

    def document_kernel(self, vec, idx):
        '''
        A method to parse documents in parallel.
        '''
        if self.reversed_wmap:
            document = [
                self.reversed_wmap[x] for x in vec.tolist()
                if x > 0 and x in self.reversed_wmap.keys()
            ]
        else:
            document = vec.split()
        hypernyms = []
        local_graph = []
        initial_hypernyms = []
        # parallel document walk
        for ix, word in enumerate(document):
            if len(word) < 2:
                continue
            wsyn = self.disambiguation_synset(word, document, ix)
            if wsyn is not None:
                initial_hypernyms.append(wsyn.name())
                paths = wsyn.hypernym_paths()
                for path in paths:
                    parent = None
                    for en, x in enumerate(path):
                        if en > self.start_term_depth:
                            if parent is not None:
                                # add hypernyms to global taxonomy
                                local_graph.append((parent.name(), x.name()))

                                # add parent, as well as current hypernym to
                                # the local tree.
                                hypernyms.append(parent.name())
                                hypernyms.append(x.name())
                                parent = x

                        else:
                            parent = x
        initial_hypernyms = set(initial_hypernyms)
        return (initial_hypernyms, idx, hypernyms, local_graph)

    def wordnet_features(self, data, wmap=None):
        """
        Construct word wector maps and use graph-based properties to select relevant number of features..
        """

        # store sequences for further transformations
        self.doc_seqs = data
        self.monitor("Constructing semantic vectors of size: {}".format(
            self.max_features))
        if wmap:
            self.reversed_wmap = {v: k for k, v in wmap.items()}

        if wmap:
            len(wmap.keys())
        self.WN = nx.DiGraph()
        self.all_hypernyms = []
        self.doc_synsets = defaultdict(list)
        if self.parallel:
            self.monitor(
                "Processing {} documents in parallel batches..".format(
                    len(self.doc_seqs)))
            jobs = [(vec, idx) for idx, vec in enumerate(self.doc_seqs)]
            self.num_cpu
            with multiprocessing.Pool(processes=self.num_cpu) as pool:
                results = pool.starmap(self.document_kernel, jobs)
                self.monitor("Constructing local taxonomy..")

                for result in tqdm(results):
                    initial_hyp, idx, hypernyms, graph = result
                    self.all_hypernyms.append(set(hypernyms))
                    self.initial_terms.append(initial_hyp)
                    self.WN.add_edges_from(graph)
                    self.doc_synsets[idx] = hypernyms
        else:
            for enx, vec in enumerate(self.doc_seqs):
                if enx % 1000 == 0:
                    self.monitor("Processed {} documents..".format(enx))
                if self.reversed_wmap:
                    document = [
                        self.reversed_wmap[x] for x in vec.tolist()
                        if x > 0 and x in self.reversed_wmap.keys()
                    ]
                else:
                    document = vec
                hypernyms = []
                for ix, word in enumerate(document):
                    wsyn = self.disambiguation_synset(word, document, ix)
                    self.initial_terms.append(wsyn)
                    if wsyn is not None:
                        paths = wsyn.hypernym_paths()
                        for path in paths:
                            parent = None
                            for x in path:
                                if parent is not None:
                                    self.WN.add_edge(parent.name(), x.name())
                                    hypernyms.append(parent.name())
                                    hypernyms.append(x.name())
                                    parent = x
                                else:
                                    parent = x

                self.all_hypernyms.append(set(hypernyms))
                self.doc_synsets[enx] = hypernyms
        self.choose_feature_selection_heuristic()

    def choose_feature_selection_heuristic(self):
        self.monitor("Selecting semantic terms..")
        if self.heuristic == "closeness_centrality":
            self.heuristic_closeness()

        if self.heuristic == "betweenness_centrality":
            self.heuristic_betweenness()

        elif self.heuristic == "rarest_terms":
            self.heuristic_specificity()

        elif self.heuristic == "mutual_info":
            self.heuristic_mutual_info()

        elif self.heuristic == "pagerank":
            self.heuristic_pagerank_selection()

        else:
            self.monitor(
                "Please select one of the following heuristics: {}".format(
                    "\n".join(self.possible_heuristics)))

    def map_data_to_digraph(self, path):
        graph = nx.DiGraph()
        with open(path, 'r') as file:
            for line in file:
                splitted = re.split(r'\t+', line)
                instance, _, concept = splitted
                graph.add_edge(instance, concept[:-1])
        return graph

    def add_all_hypernyms(self, node):
        token = str(node).lower()
        hypernyms = []

        if self.knowledge_base.has_node(token):
            for key in [h for h in self.knowledge_base.successors(token)]:
                hypernyms.append(key)
        else:
            return None
        return hypernyms

    def add_best_n_hypernyms(self, node, n):
        token = str(node).lower()
        hypernyms = []
        if self.knowledge_base.has_node(token):
            edges = self.knowledge_base.edges(token)
            if len(edges) > 0:
                assert n > 0
                i = 0
                for u, v in edges:
                    if i == n:  # edges are ordered, so we can choose the first one
                        break
                    hypernyms.append(v)
                    i += 1
            return hypernyms
        return None

    def one_document_hypernyms(self, vec, idx, hypernyms_count):
        local_graph = []
        initial_terms = []
        hypernyms = []

        if self.reversed_wmap:
            document = [
                self.reversed_wmap[x] for x in vec.tolist()
                if x > 0 and x in self.reversed_wmap.keys()
            ]
        else:
            document = vec

        text = ' '.join(document)
        doc = self.nlp(text)

        for token in doc:
            if token.tag_ == 'NNP' or token.tag_ == 'NN':
                if str(token) == self.document_split_symbol:
                    continue
                initial_terms.append(str(token))

                if hypernyms_count == 'all':  # add all hypernyms of given word
                    out = self.add_all_hypernyms(token)
                elif isinstance(hypernyms_count,
                                int):  # add just the best hypernym
                    out = self.add_best_n_hypernyms(token, hypernyms_count)
                else:
                    print(
                        "Enter either positive integer or 'all' as the parameter 'hyp'."
                    )
                    return

                if out is not None:
                    hypernyms.extend(out)
                    for h in out:
                        local_graph.append((str(token), h))

        return initial_terms, idx, hypernyms, local_graph

    def knowledge_graph_features(self, data, wmap=None, hyp=1, path=None):
        """
        Uses Microsoft Concet Graph for finding hypernyms
        :param data:
        :param wmap:
        :param hypernyms: "all" or positive integer
        :param path: path to file with relations from knowledge graph
        :return:
        """
        self.knowledge_graph = True
        self.doc_seqs = data
        self.all_hypernyms = []
        self.initial_terms = []
        self.doc_synsets = defaultdict(list)
        self.WN = nx.DiGraph()

        if path is None:
            self.monitor(
                "Please, enter a name of path to knowledge graph source")
            return
        else:
            self.knowledge_base = self.map_data_to_digraph(path)

        if wmap:
            self.reversed_wmap = {v: k for k, v in wmap.items()}
        self.monitor("Looking up hypernyms using knowledge graph")

        for enx, vec in enumerate(self.doc_seqs):
            result = self.one_document_hypernyms(vec, enx, hyp)
            initial_terms, idx, hypernyms, graph = result
            self.initial_terms.append(list(initial_terms))
            self.doc_synsets[idx] = hypernyms
            self.WN.add_edges_from(graph)
            self.all_hypernyms.append(set(hypernyms))

        self.choose_feature_selection_heuristic()

    def heuristic_pagerank_selection(self):
        '''
        Personalized PageRank for term prioretization -> what is relevant with respect to hypernym mappings?
        '''

        personalization = None
        # if not self.knowledge_graph:
        self.initial_terms = set().union(*[set(x) for x in self.initial_terms])
        personalization = {}
        for node in self.WN.nodes():
            if node in self.initial_terms:
                personalization[node] = 1
            else:
                personalization[node] = 0

        prC = nx.pagerank(self.WN, personalization=personalization)
        self.semantic_candidates = []
        self.pagerank_scores = []
        for en, k in enumerate(sorted(prC, key=prC.get, reverse=True)):
            if en < self.max_features:
                self.semantic_candidates.append(k)
                self.pagerank_scores.append(prC[k])
            else:
                break

    def heuristic_closeness(self):
        '''
        Closeness centrality method
        '''

        eigC = nx.closeness_centrality(self.WN)
        self.semantic_candidates = []
        for en, k in enumerate(sorted(eigC, key=eigC.get, reverse=False)):
            if en < self.max_features:
                self.semantic_candidates.append(k)
            else:
                break

    def heuristic_betweenness(self):
        '''
        Closeness centrality method
        '''

        eigC = nx.betweenness_centrality(self.WN)
        self.semantic_candidates = []
        for en, k in enumerate(sorted(eigC, key=eigC.get, reverse=False)):
            if en < self.max_features:
                self.semantic_candidates.append(k)
            else:
                break

    def heuristic_specificity(self):
        '''
        Simple term count-sort combination
        '''

        lot = [list(x) for x in self.all_hypernyms]
        flat_list = [item for sublist in lot for item in sublist]
        rarest_terms = Counter(flat_list)
        self.semantic_candidates = [
            x[0] for x in sorted(rarest_terms.items(),
                                 key=operator.itemgetter(1),
                                 reverse=False)
        ][0:self.max_features]
        if self.hypernym_space is not None:
            try:
                self.monitor("Saving hypernym distribution")
                hyper_subs = np.array(list(rarest_terms.values()))
                np.save(self.hypernym_space, hyper_subs)
            except BaseException:
                self.monitor("Folder structure insufficient!")

    def feature_transform(self, matrix):
        '''
        Helper method for transformation of the target matrix given a set of features.
        '''

        if self.indices_selected_features is None:
            raise ValueError(
                "Semantic candidates not yet defined. Please, run the heuristic_mutual_info step first."
            )
        return matrix[:, self.indices_selected_features]

    def heuristic_mutual_info(self):
        '''
        Mutual information heuristic
        '''

        self.semantic_candidates = list(
            set([
                item for sublist in [list(x) for x in self.all_hypernyms]
                for item in sublist
            ]))

        # potential feature selection
        if self.targets is not None:
            tmp = self.transform()
            self.monitor("Finding the best scoring target..")

            # select optimal splitting target
            mutual_info_scores = []
            # transform to one-hot encoding if needed
            if np.ndim(self.targets) == 1:
                n = len(self.targets)
                onehot = np.zeros((n, max(self.targets) + 1))
                onehot[np.arange(n), self.targets] = 1
                self.targets = onehot
            for j in range(self.targets.shape[1]):
                mutual_info_tmp = mutual_info_classif(np.rint(tmp),
                                                      self.targets[:, j])
                mutual_info_scores.append(mutual_info_tmp)
            sum_scores = np.zeros(tmp.shape[1])
            for score in mutual_info_scores:
                sum_scores += score
            sum_scores = sum_scores / self.targets.shape[1]
            if self.indices_selected_features is None:
                # take top n by score as candidates
                self.indices_selected_features = sum_scores.argsort(
                )[-self.max_features:]
                self.top_mutual_information_scores = np.sort(
                    sum_scores)[-self.max_features:]

            if self.class_names is not None:
                self.relevant_classes = []
                for el_idx in self.indices_selected_features:
                    ranks = {}
                    for enx, vec in enumerate(mutual_info_scores):
                        ranks[enx] = vec[el_idx]
                    sorted_ranks = sorted(ranks.items(),
                                          key=operator.itemgetter(1),
                                          reverse=True)
                    sorted_outranks = []
                    for rank in sorted_ranks:
                        sorted_outranks.append(
                            (self.class_names[rank[0]], rank[1]))

                    # append whole array of priorities.
                    self.relevant_classes.append(sorted_outranks)

            # update the actual semantic candidates..
            self.semantic_candidates = [
                j for e, j in enumerate(self.semantic_candidates)
                if e in self.indices_selected_features
            ]

            #            self.initial_transformation = True
            self.skip_transform = True
            self.transformed_matrix = self.feature_transform(tmp)

    def feature_kernel(self, ix):
        '''
        Paralel computation of doc-feature values
        '''

        local_hyps = set(self.doc_synsets[ix])
        pnts = []
        for idx, x in enumerate(self.semantic_candidates):
            if x in local_hyps:
                local = Counter(self.doc_synsets[ix])
                lfreq = local[x]
                lmax = local[max(local, key=local.get)]
                lidf = len(self.all_hypernyms) / \
                    len([j for j in self.all_hypernyms if x in j])
                # average weighted tfidf
                weight = (0.5 + 0.5 * (lfreq / (lmax + 1))) * np.log(lidf)
                pnts.append((idx, ix, weight))
        return pnts

    def transform(self, docs=None):
        '''
        The generic transform method.
        '''

        if self.skip_transform:
            self.skip_transform = False
            return self.transformed_matrix

        # generate the semantic mappings
        if docs is None:
            self.tmp_doc_seqs = self.doc_seqs
        else:
            self.tmp_doc_seqs = self.tokenizer.texts_to_sequences(docs)

        if self.heuristic == "mutual_info":
            rows, cols = len(self.tmp_doc_seqs), len(self.semantic_candidates)
            nsem = len(self.semantic_candidates)
        else:
            rows, cols = len(self.tmp_doc_seqs), self.max_features
            nsem = self.max_features

        print("... Computing weights for {} semantic vectors ...".format(nsem))

        rs = []
        cs = []
        data = []

        if self.parallel:
            self.num_cpu
            jobs = list(range(len(self.tmp_doc_seqs)))
            with multiprocessing.Pool(processes=self.num_cpu) as pool:
                results = tqdm(pool.imap(self.feature_kernel, jobs),
                               total=len(jobs))
                results = [x for y in results for x in y]
                cs, rs, data = zip(*results)
        else:
            for ix, _ in enumerate(self.tmp_doc_seqs):
                local_hyps = set(self.doc_synsets[ix])
                for idx, x in enumerate(self.semantic_candidates):
                    if x in local_hyps:
                        # compute weighted tfidf
                        local = Counter(self.doc_synsets[ix])
                        lfreq = local[x]
                        lmax = local[max(local, key=local.get)]
                        lidf = len(self.all_hypernyms) / \
                            len([j for j in self.all_hypernyms if x in j])
                        # average weighted tfidf
                        weight = (0.5 + 0.5 * (lfreq / (lmax + 1))) * \
                            np.log(lidf)
                        cs.append(idx)
                        rs.append(ix)
                        data.append(weight)  # this can be arbitrary weight!

        assert len(rs) == len(cs)
        m = sps.csr_matrix((data, (rs, cs)), shape=(rows, cols))
        if self.initial_transformation:
            return self.feature_transform(m)
        else:
            return m

    def fit_transform(self, text):
        '''
        The generic fit-transform combination.
        '''

        self.monitor("Constructing the taxonomy tree..")
        self.fit(text)
        self.monitor("Constructing feature vectors..")
        return self.transform()
