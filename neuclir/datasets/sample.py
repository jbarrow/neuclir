from typing import List, Tuple, Callable, Dict, Set, Any, Union
from allennlp.data.tokenizers.word_filter import StopwordFilter
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from collections import defaultdict
from scipy.special import logsumexp
from scipy.misc import logsumexp
from functools import reduce
from shutil import copyfile
from tqdm import tqdm

import pandas as pd
import argparse
import logging
import random
import glob
import json
import sys
import os


Path = str
DocumentID = str
QueryID = str

def dict_from_paths(paths: List[Path], strip_file: bool = True) -> Dict[str, Path]:
    output = {}
    for path in paths:
        print(path, ': ', len(glob.glob(path)))
        for f in glob.glob(path):
            filename = os.path.basename(f)
            #filename = os.path.splitext(filename)[0] if strip_file else filename
            filename = filename.split('.')[0] if strip_file else filename
            output[filename] = f
    return output

def docs_from_paths(paths: List[Path], strip_file: bool = True, tokenizer: Tokenizer = WordTokenizer()) -> Dict[str, List[str]]:
    docs = dict_from_paths(paths, strip_file)
    texts = []

    for doc, path in tqdm(docs.items()):
        with open(path) as fp:
            texts.append((doc, fp.read()))

    tokenized = tokenizer.batch_tokenize([t[1] for t in texts])
    tokenized = [[w.orth_ for w in t] for t in tokenized]

    return {doc: tokens for (doc, _), tokens in zip(texts, tokenized)}

def json_read_tokens(filename: Path, flatten: bool = True) -> List[str]:
    tokens = []
    with open(filename) as fp:
        for line in fp:
            l = json.loads(line)
            ts = [w['word'].lower() for w in l[0]] if len(l) > 0 else []
            tokens.extend(ts) if flatten else tokens.append(ts)
    return tokens

def json_docs_from_paths(paths: List[Path], strip_file: bool = True) -> Dict[str, List[str]]:
    docs = dict_from_paths(paths, strip_file)
    return { doc: json_read_tokens(path) for doc, path in tqdm(docs.items())}

# normalization functions
def flatten(l_of_ls):
    return [i for l in l_of_ls for i in l]

def sto_normalization(df: pd.DataFrame) -> pd.DataFrame:
    if df.score.values.dtype != 'float64':
        return df
    df.score = df.score.values - logsumexp(df.score.values)
    return df

def zero_one_normalization(df: pd.DataFrame) -> pd.DataFrame:
    if df.score.values.dtype != 'float64':
        return df
    df.score = df.score.values - min(df.score.values)
    if max(df.score.values) > 0:
        df.score = df.score.values / max(df.score.values)
    return df

def no_normalization(df: pd.DataFrame) -> pd.DataFrame:
    return df

class DatasetGenerator(object):
    def __init__(self,
                 name: str,
                 params: Dict[str, Any],
                 output: Path, systems: List[str] = [],
                 f_normalization: Callable[[pd.DataFrame], pd.DataFrame] = no_normalization) -> None:
        self.params = params
        self.output = output
        self.name = name
        self.systems = systems
        self.f_normalization = f_normalization

    def read_scores_file(self, file: Path) -> Tuple[str, pd.DataFrame]:
        """
        Read a .trec file for a specific query and return the dataframe.
        """

        try:
            # load the tsv file
            df = pd.read_csv(file, sep='\t', header=None,
                names=['query_id', 'Q0', 'document_id', 'rank', 'score', 'system'],
                usecols=['document_id', 'score'])
            #df = pd.read_csv(file, sep='\t')
            # get the query id from the header
            q_id = os.path.splitext(os.path.basename(file))[0][2:]
            #df.query_id.ix[0]
            #q_id = pd.columns.values[0]
            #df.columns = ['document_id', 'score']

            return q_id, df
        except pd.errors.EmptyDataError:
            logging.log(logging.WARN, f' * No matcher results for {file}')
            return '', pd.DataFrame()

    def read_scores(self, systems: Dict[str, Union[Path, List[Path]]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Return a dict that maps the query id to the system and dataframe containing the
        document-level scores for each system.
        """
        scores = defaultdict(dict)

        for system, paths in systems.items():
            if type(paths) != list:
                paths = [paths]
            for path in paths:
                logging.log(logging.INFO, f'  - Reading in scores for "{system}" from {path}')
                score_dfs = [self.read_scores_file(qf) for qf in glob.glob(os.path.join(path, '*.trec'))]
                for query, score_df in score_dfs:
                    if system not in scores[query]:
                        scores[query][system] = score_df
                    else:
                        scores[query][system] = pd.concat([scores[query][system], score_df])
        return scores

    def read_query(self, filename: Path, query_type: str = 'words') -> List[str]:
        with open(filename) as fp:
            query = json.load(fp)
        return query['english'][query_type].strip().split()

    def read_judgments(self, file: Path) -> pd.DataFrame:
        logging.log(logging.INFO, f'  - Reading in judgments from {file}')
        judgments = pd.read_csv(file, sep='\t')
        judgments['relevant'] = 1
        return judgments

    def get_scores(self, scores: Dict[str, pd.DataFrame], doc_id: str) -> Dict[str, float]:
        all_scores = []
        for system in self.systems:
            df = scores[system]

            found = df[df.document_id == doc_id]
            score = 0.
            if len(found) == 0 and len(df) > 0:
                score = min(df.score)
            elif len(found) > 0:
                score = found.score.values[0]
            all_scores.append({'system': system, 'score': score})
        return all_scores

    def get_relevant(self, df: pd.DataFrame, doc_id: str) -> int:
        found = df[df.doc_id == doc_id]
        if len(found) == 0:
            return 0
        return 1

    def prepare(self):
        logging.log(logging.INFO, 'Loading in data')

        logging.log(logging.INFO, ' * Creating directory ' + self.output)
        try:
            os.mkdir(self.output)
        except FileExistsError:
            logging.log(logging.INFO, ' * Directory already exists, overwriting previous data.')

        # get the system scores
        logging.log(logging.INFO, ' * Reading in score files')
        self.scores = self.read_scores(self.params['scores'])
        self.scores = self.normalize(self.scores)

        if 'judgments' in self.params:
            # get the relevance judgments
            logging.log(logging.INFO, ' * Reading in relevance judgments')
            judgement_files = self.params.pop('judgments', [])
            self.judgments = pd.concat([self.read_judgments(f) for f in judgement_files])
            # get the queries with judgments
            self.queries_with_judgments = pd.DataFrame({'query_id': sorted(set(self.judgments.query_id))})
            logging.log(logging.INFO, f' * There are {len(self.queries_with_judgments)} total queries with judgments')
        else:
            logging.log(logging.INFO, ' * Skipping relevance judgments')
            self.judgments = None
            self.queries_with_judgments = None

        if 'queries' in self.params:
            # get all the queries
            self.queries = dict_from_paths(self.params['queries'])
            logging.log(logging.INFO, f' * Found {len(self.queries)} total queries')
        elif 'query' in self.params:
            self.query = self.params['query']
        # load in the possible documents
        if 'filter_stopwords' in self.params and self.params['filter_stopwords']:
            sw_filter = StopwordFilter()
            tokenizer = WordTokenizer(word_filter=sw_filter)
        else:
            tokenizer = WordTokenizer()

        if 'doctype' in self.params and self.params['doctype'] == 'json':
            self.docs = json_docs_from_paths(self.params['docs'])
        else:
            self.docs = docs_from_paths(self.params['docs'], tokenizer=tokenizer)
        logging.log(logging.INFO, f' * Found {len(self.docs)} total docs')
        # # impose an order on the systems
        # self.systems = sorted(self.params['scores'].keys())

    def normalize(self, queries: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        normalized = {}

        for query, scores in queries.items():
            normalized[query] = {}
            for system, df in scores.items():
                normalized[query][system] = self.f_normalization(df)

        return normalized

    def sample_dataset(self):
        pass

def s_random(docs: Set[str], n: int = 1) -> List[str]:
    return random.sample(docs, n)

def s_difficult(df: pd.DataFrame, n: int = 1) -> List[str]:
    return  [(df.iloc[sample].document_id) for sample in random.sample(set(range(len(df))), n)]

class PairedDatasetGenerator(DatasetGenerator):
    _dispatch = {
        'random': s_random,
        'difficult': s_difficult,
    #    'pooled_plus': sample_pooled_plus
    }

    def sample_dataset(self) -> None:
        logging.log(logging.INFO, f'Creating PAIRED dataset: {self.name}')
        # load all the scores, judgments, and data files
        self.prepare()
        # open the output file and begin sampling the dataset
        out_fp = open(os.path.join(self.output, f'{self.name}.json'), 'w')
        # when generating a training set, we want to associate each relevant
        # (query, doc) pair with one or more irrelevant (query, doc) pairs.
        with tqdm(total=len(self.judgments)) as pbar:
            for query, df in self.judgments.groupby('query_id'):
                if query not in self.scores:
                    continue
                query_text = self.read_query(self.queries[query])
                query_scores = self.scores[query]
                # if we choose a sampling strategy that requires it
                possible = set(self.docs.keys()) - set(df.doc_id)
                if self.params['strategy'] == 'difficult' and random.random() <= self.params['p_difficult']:
                    # exclude relevant docs
                    tmp_possible = set(query_scores[self.params['sample_system']].document_id) - set(df.doc_id)
                    if len(tmp_possible) >= self.params['n_irrelevant']:
                        possible = tmp_possible
                # sample irrelevant docs for every relevant doc
                for judgement in df.itertuples():
                    n_irrelevant = self.params['n_irrelevant']
                    # sample a position to put the relevant document in
                    position = random.randint(0, n_irrelevant)
                    #position = [random.randint(0, 1) for i in range(self.params['n_irrelevant'])]
                    # sample paired documents
                    sampled_docs = s_random(possible, n_irrelevant)
                    # if strategy == 'difficult' and len(possible) > n_irrelevant:
                    #     sampled_docs = s_difficult(possible, n_irrelevant)
                    # else:
                    #     sampled_docs = s_random(possible, n_irrelevant)
                    # get the tokens for the documents
                    relevant = {
                        #'text': ' '.join(self.read_tokens(self.docs[judgement.doc_id], is_json=False)),
                        'text': ' '.join(self.docs[judgement.doc_id]),
                        'scores': self.get_scores(query_scores, judgement.doc_id)
                    }
                    # get the tokens for the irrelevant documents
                    irrelevant = [{
                        #'text': ' '.join(self.read_tokens(self.docs[doc_id], is_json=False)),
                        'text': ' '.join(self.docs[doc_id]),
                        'scores': self.get_scores(query_scores, doc_id)
                    } for doc_id in sampled_docs]

                    #for p, i in zip(position, irrelevant):
                    #   ordered_docs = [i, relevant] if p == 1 else [relevant, i]
                    # generate the output json line
                    outline = json.dumps({
                        'query': ' '.join(query_text),
                        'docs': irrelevant[:position] + [relevant] + irrelevant[position:],
                        'relevant': position
                        #'docs': ordered_docs,
                        #'relevant': p
                    })
                    # write the line to the json file
                    out_fp.write(outline + '\n')

                    pbar.update(1)
        out_fp.close()

class RerankingDatasetGenerator(DatasetGenerator):
    def has_relevant_docs(self, scores: Dict[str, pd.DataFrame]) -> bool:
        for system, df in scores.items():
            if len(df) > 0 and system in self.systems:
                return True
        return False

    def sample_dataset(self) -> None:
        logging.log(logging.INFO, f'Creating RERANKING dataset: {self.name}')
        # load all the datasets, score files, etc.
        self.prepare()
        # open the file
        dataset_fp = open(os.path.join(self.output, f'{self.name}.json'), 'w')
        scoring_fp = open(os.path.join(self.output, f'{self.name}_scoring.json'), 'w')
        # when generating a validation set, our goal is to rerank, so we want
        # to load in all scores for a given query, up to, say, 100.
        for query, dfs in tqdm(self.scores.items()):
            #print(query, dfs.keys())
            query_text = self.read_query(self.queries[query])
            relevant = self.judgments[self.judgments['query_id'] == query]
            # compute the number of relevant docs that we missed

            if self.has_relevant_docs(dfs):
                # get the set of all documents returned by the systems
                docs = set(pd.concat([df for system, df in dfs.items() if system in self.systems]).document_id)
                ignored_relevant = len(set(relevant.doc_id) - docs)
                ignored_irrelevant = len(self.docs) - len(docs) - ignored_relevant
                scored_docs = []

                for doc in docs:
                    scored_docs.append({
                        'document_id': doc,
                        #'text': ' '.join(self.read_tokens(self.docs[doc], is_json=False)),
                        'text': ' '.join(self.docs[doc]),
                        'scores': self.get_scores(dfs, doc),
                        'relevant': self.get_relevant(relevant, doc)
                    })
                # write a single query
                outline = json.dumps({
                    'query_id': query,
                    'query': ' '.join(query_text),
                    'docs': scored_docs,
                    'ignored_relevant': ignored_relevant,
                    'ignored_irrelevant': ignored_irrelevant
                })
                dataset_fp.write(outline + '\n')
            else:
                ignored_relevant = len(relevant)
                ignored_irrelevant = len(self.docs) - ignored_relevant

                outline = json.dumps({
                    'query_id': query,
                    'ignored_relevant': ignored_relevant,
                    'ignored_irrelevant': ignored_irrelevant
                })

                scoring_fp.write(outline + '\n')

        dataset_fp.close()
        scoring_fp.close()


class RankingDatasetGenerator(DatasetGenerator):
    def sample_dataset(self) -> None:
        logging.log(logging.INFO, f'Creating RANKING dataset: {self.name}')
        # load all the datasets, score files, etc.
        self.prepare()
        # open the file
        dataset_fp = open(os.path.join(self.output, f'{self.name}.json'), 'w')
        # when generating a validation set, our goal is to rerank, so we want
        # to load in all scores for a given query, up to, say, 100.
        for query, path in tqdm(self.queries.items()):
            dfs = self.scores.get(query, {})
            query_text = self.read_query(path)
            relevant = self.judgments[self.judgments['query_id'] == query]

            scored_docs = []
            for doc_id, doc in self.docs.items():
                scored_docs.append({
                    'document_id': doc_id,
                    #'text': ' '.join(self.read_tokens(path, is_json=False)),
                    'text': ' '.join(doc),
                    'scores': self.get_scores(dfs, doc_id),
                    'relevant': self.get_relevant(relevant, doc_id)
                })
            # write a single query
            outline = json.dumps({
                'query_id': query,
                'query': ' '.join(query_text),
                'docs': scored_docs
            })
            dataset_fp.write(outline + '\n')

        dataset_fp.close()


_dataset_dispatch = {
    'paired': PairedDatasetGenerator,
    'reranking': RerankingDatasetGenerator,
    'ranking': RankingDatasetGenerator
}

_normalization_dispatch = {
    'sto': sto_normalization,
    'none': no_normalization,
    'zero_one': zero_one_normalization
}

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage:\npython -m neuclir.datasets.sample [CONFIG]')
        sys.exit(-1)


    settings = json.load(open(sys.argv[1]))
    # set up the logger
    logging_level = settings.pop('logging', 'info').upper()
    logging.basicConfig(level=logging.getLevelName(logging_level))

    f_normalization = _normalization_dispatch[settings['normalization']]

    for name, dataset in settings['datasets'].items():
        generator = _dataset_dispatch[dataset['type']](name, dataset, settings['output'], settings['systems'], f_normalization)
        generator.sample_dataset()

    # copy over the selected files
    copyfile(sys.argv[1], os.path.join(settings['output'], os.path.basename(sys.argv[1])))
