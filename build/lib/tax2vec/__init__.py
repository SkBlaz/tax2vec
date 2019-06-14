
##############################
## tax2vec --- Blaz Skrlj 2019
##############################

try:
    from tqdm import tqdm
    pbar = True
except:
    pbar = False    
    pass
import multiprocessing
from collections import defaultdict
import pickle
import numpy as np
try:
    import nltk
    nltk.data.path.append("./nltk_data")
except Exception as es:
    print(es)
    
from nltk.wsd import lesk
import itertools
import multiprocessing as mp
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn import preprocessing
from nltk.corpus import wordnet as wn
import networkx as nx
from collections import Counter
import itertools
import operator
import scipy.sparse as sps
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

class tax2vec:
    def __init__(self, max_features=500, disambiguation_window=10,document_split_symbol="mergertag",heuristic="mutual_info", num_cpu=1,hypernym_distribution = "./hypernym_space/dist1.npy",targets=None,class_names=None, start_term_depth = 0):

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
        self.initial_terms = [] ## for personalized node ranking
        self.possible_heuristics = ["closeness_centrality","rarest_terms","mutual_info","pagerank"]
        self.doc_seqs = None
        self.parallel = True
        
        if num_cpu == "all":
            self.num_cpu = mp.cpu_count()
            
        elif num_cpu == 1:
            self.parallel = False
#            self.num_cpu = 1
            
        else:
            self.num_cpu=num_cpu
        self.monitor("Using {} heuristic.".format(self.heuristic))
        
    def monitor(self,message):
        '''
        A very simple state monitor
        '''

        logging.info(message)
        
    def fit(self,data,wmap):
        '''
        A simple fit method
        '''        
        self.monitor("Constructing local taxonomy...")
        self.wordnet_features(data,wmap)
        
    def disambiguation_synset(self, word, document, word_index=None):

        '''
        First split each multidocument and select most frequent hypernyms in terms of occurrence.
        '''
        
        if self.disambiguation_window is None:
            
            ## take into account whole docs
            docs = " ".join(document).split(self.document_split_symbol)
            hyps = []
            for doc in docs:
                ww = " ".join(doc)
                hypernym = lesk(ww, word)
                if hypernym is not None:
                    hyps.append(hypernym)
            if len(hyps) > 0:
                cntr = Counter(hyps)
                return max(set(hyps), key = hyps.count)
            else:
                return None
        
        else:
            if (word_index-self.disambiguation_window) > 0:
                bottom_index = (word_index-self.disambiguation_window)
            else:
                bottom_index = 0

            if (word_index+self.disambiguation_window) > len(document):
                top_index = len(document)
            else:
                top_index = (word_index+self.disambiguation_window)

            word_window = document[bottom_index:top_index]
            ww = " ".join(word_window)
            hypernym = lesk(ww,word)

            if hypernym is not None:
                return hypernym
            else:
                return None


    def document_kernel(self,vec,idx):

        '''
        A method to parse documents in parallel.
        '''
        
        document = [self.reversed_wmap[x] for x in vec.tolist() if x > 0 and x in self.reversed_wmap.keys()]        
        hypernyms = []
        local_graph = []
        initial_hypernyms = []
        ## parallel document walk
        for ix,word in enumerate(document):
            if len(word) < 2:
                continue
            wsyn = self.disambiguation_synset(word, document, ix)
            if wsyn is not None:
                initial_hypernyms.append(wsyn.name())
                paths = wsyn.hypernym_paths()
                common_terms = []
                for path in paths:
                    parent = None
                    for en, x in enumerate(path):
                        if en > self.start_term_depth:
                            if parent is not None:

                                ## add hypernyms to global taxonomy
                                local_graph.append((parent.name(),x.name()))

                                ## add parent, as well as current hypernym to the local tree.
                                hypernyms.append(parent.name())
                                hypernyms.append(x.name())
                                parent = x

                        else:
                            parent = x
        initial_hypernyms = set(initial_hypernyms)
        return (initial_hypernyms,idx,hypernyms,local_graph)
            
    def wordnet_features(self,data, wmap):

        """
        Construct word wector maps and use graph-based properties to select relevant number of features..
        """
        
        ## store sequences for further transformations
        self.doc_seqs = data        
        self.monitor("Constructing semantic vectors of size: {}".format(self.max_features))
        self.reversed_wmap = {v : k for k,v in wmap.items()}

        flist = []
        tw = len(wmap.keys())
        ldct = {}
        cnts = {}        
        processed = 0        
        self.WN = nx.DiGraph()
        total_wordset = {}
        self.all_hypernyms = []
        self.doc_synsets = defaultdict(list)
        if self.parallel:
            self.monitor("Processing {} documents in parallel batches..".format(len(self.doc_seqs)))
            jobs = [(vec,idx) for idx,vec in enumerate(self.doc_seqs)]
            num_cpu = self.num_cpu
            with multiprocessing.Pool(processes=self.num_cpu) as pool:
                results = pool.starmap(self.document_kernel, jobs)
                self.monitor("Constructing local taxonomy..")
                all_edges = []

                for result in tqdm(results):
                    initial_hyp,idx,hypernyms,graph = result
                    self.all_hypernyms.append(set(hypernyms))
                    self.initial_terms.append(initial_hyp)
                    self.WN.add_edges_from(graph)
                    self.doc_synsets[idx] = hypernyms
        else:
            for enx, vec in enumerate(self.doc_seqs):
                if enx % 1000 == 0:
                    self.monitor("Processed {} documents..".format(enx))
                document = [self.reversed_wmap[x] for x in vec.tolist() if x > 0 and x in self.reversed_wmap.keys()]
                hypernyms = []
                for ix,word in enumerate(document):
                    wsyn = self.disambiguation_synset(word, document, ix)
                    self.initial_terms.append(wsyn)
                    if wsyn is not None:       
                        paths = wsyn.hypernym_paths()
                        common_terms = []
                        for path in paths:
                            parent = None
                            for x in path:
                                if parent is not None:
                                    self.WN.add_edge(parent.name(),x.name())
                                    hypernyms.append(parent.name())
                                    hypernyms.append(x.name())
                                    parent = x
                                else:
                                    parent = x

                self.all_hypernyms.append(set(hypernyms))
                self.doc_synsets[enx] = hypernyms

        self.monitor("Selecting semantic terms..")
        if self.heuristic == "closeness_centrality":
            self.heuristic_closeness()

        elif self.heuristic == "rarest_terms":
            self.heuristic_specificity()

        elif self.heuristic == "mutual_info":
            self.heuristic_mutual_info()

        elif self.heuristic == "pagerank":
            self.heuristic_pagerank_selection()
            
        else:
            self.monitor("Please select one of the following heuristics: {}".format("\n".join(self.possible_heuristics)))

    def heuristic_pagerank_selection(self):

        '''
        Personalized PageRank for term prioretization -> what is relevant with respect to hypernym mappings?
        '''
        
        personalization = None
        self.initial_terms = set().union(*[set(x) for x in self.initial_terms])
        personalization = {}
        for node in self.WN.nodes():
            if node in self.initial_terms:
                personalization[node] = 1
            else:
                personalization[node] = 0
                
        prC = nx.pagerank(self.WN,personalization=personalization)
        self.semantic_candidates = []
        self.pagerank_scores = []
        for en, k in enumerate(sorted(prC, key=prC.get,reverse=True)):
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
        for en, k in enumerate(sorted(eigC, key=eigC.get,reverse=False)):
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
        self.semantic_candidates = [x[0] for x in sorted(rarest_terms.items(), key=operator.itemgetter(1),reverse=False)][0:self.max_features]
        if self.hypernym_space is not None:
            try:
                self.monitor("Saving hypernym distribution")
                hyper_subs = np.array(list(rarest_terms.values()))
                np.save(self.hypernym_space,hyper_subs)
            except:
                self.monitor("Folder structure insufficient!")

    def feature_transform(self, matrix):

        '''
        Helper method for transformation of the target matrix given a set of features.
        '''
        
        if self.indices_selected_features is None:
            raise ValueError("Semantic candidates not yet defined. Please, run the heuristic_mutual_info step first.")
        return matrix[:,self.indices_selected_features]        
        
    def heuristic_mutual_info(self):

        '''
        Mutual information heuristic
        '''
        
        self.semantic_candidates = list(set([item for sublist in [list(x) for x in self.all_hypernyms] for item in sublist]))

        ## potential feature selection
        if self.targets is not None:
            tmp = self.transform()            
            self.monitor("Finding the best scoring target..")
            
            ## select optimal splitting target
            mutual_info_scores = []
            for j in range(self.targets.shape[1]):
                mutual_info_tmp = mutual_info_classif(np.rint(tmp),self.targets[:,j])
                mutual_info_scores.append(mutual_info_tmp)                
            sum_scores = np.zeros(tmp.shape[1])
            for score in mutual_info_scores:
                sum_scores+=score
            sum_scores = sum_scores/self.targets.shape[1]
            if self.indices_selected_features is None:
                
                ## take top n by score as candidates
                self.indices_selected_features = sum_scores.argsort()[-self.max_features:]
                self.top_mutual_information_scores = np.sort(sum_scores)[-self.max_features:]

            if self.class_names is not None:
                self.relevant_classes = []
                for el_idx in self.indices_selected_features:
                    ranks = {}
                    for enx, vec in enumerate(mutual_info_scores):
                        ranks[enx] = vec[el_idx]
                    sorted_ranks = sorted(ranks.items(), key=operator.itemgetter(1),reverse=True)
                    sorted_outranks = []
                    for rank in sorted_ranks:
                        sorted_outranks.append((self.class_names[rank[0]], rank[1]))
                        
                    ## append whole array of priorities.
                    self.relevant_classes.append(sorted_outranks)
            
            ## update the actual semantic candidates..
            self.semantic_candidates = [j for e,j in enumerate(self.semantic_candidates) if e in self.indices_selected_features]
            
#            self.initial_transformation = True
            self.skip_transform = True
            self.transformed_matrix = self.feature_transform(tmp)

    def feature_kernel(self,ix):

        '''
        Paralel computation of doc-feature values
        '''
        
        local_hyps = set(self.doc_synsets[ix])
        pnts = []
        for idx,x in enumerate(self.semantic_candidates):
            if x in local_hyps:
                local = Counter(self.doc_synsets[ix])
                lfreq = local[x]
                lmax = local[max(local, key=local.get)]
                lidf = len(self.all_hypernyms)/len([j for j in self.all_hypernyms if x in j])
                ## average weighted tfidf
                weight = (0.5+0.5*(lfreq/(lmax+1))) * np.log(lidf)
                pnts.append((idx,ix,weight))
        return pnts
    
    def transform(self,doc_seqs=None):

        '''
        The generic transform method.
        '''
        
        if self.skip_transform:
            self.skip_transform = False
            return self.transformed_matrix
                
        ## generate the semantic mappings
        if doc_seqs is None:
            self.tmp_doc_seqs = self.doc_seqs
        else:
            self.tmp_doc_seqs = doc_seqs

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
            num_cpu = self.num_cpu
            jobs = list(range(len(self.tmp_doc_seqs)))
            with multiprocessing.Pool(processes=self.num_cpu) as pool:
                results = tqdm(pool.imap(self.feature_kernel, jobs),total=len(jobs))
                results = [x for y in results for x in y]
                cs,rs,data = zip(*results)
        else:
            for ix,_ in enumerate(self.tmp_doc_seqs):            
                local_hyps = set(self.doc_synsets[ix])
                for idx,x in enumerate(self.semantic_candidates):
                    if x in local_hyps:
                        
                        ## compute weighted tfidf
                        local = Counter(self.doc_synsets[ix])
                        lfreq = local[x]
                        lmax = local[max(local, key=local.get)]
                        lidf = len(self.all_hypernyms)/len([j for j in self.all_hypernyms if x in j])
                        ## average weighted tfidf
                        weight = (0.5+0.5*(lfreq/(lmax+1))) * np.log(lidf)
                        cs.append(idx)
                        rs.append(ix)
                        data.append(weight) ## this can be arbitrary weight!

        assert len(rs) == len(cs)
        m = sps.csr_matrix((data, (rs, cs)), shape=(rows,cols))
        if self.initial_transformation:
            return self.feature_transform(m)
        else:
            return m

    def fit_transform(self,data,wmap):

        '''
        The generic fit-transform combination.
        '''
        
        self.monitor("Constructing the taxonomy tree..")
        self.wordnet_features(data,wmap)
        self.monitor("Constructing feature vectors..")
        return self.transform()

