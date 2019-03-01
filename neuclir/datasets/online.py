from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_filter import StopwordFilter
from .sample import DatasetGenerator, dict_from_paths, sto_normalization
from typing import List, Dict, Any

import pandas as pd
import os

class OnlineDatasetGenerator(DatasetGenerator):
    def __init__(self, params: Dict[str, Any], systems: List[str] = []):
        self.params = params
        self.systems = systems
        self.f_normalize = sto_normalization
        #self.sw_filter = StopwordFilter()
        self.tokenizer = WordTokenizer()#(word_filter=self.sw_filter)

    def read_scores_file(self, file: str) -> pd.DataFrame:
        """
        Read a .trec file for a specific query and return the dataframe.
        """
        try:
            df = pd.read_csv(file, sep=' ', header=None,
                names=['query_id', 'Q0', 'document_id', 'rank', 'score', 'system'],
                usecols=['document_id', 'score'])
        except pd.errors.EmptyDataError:
            logging.log(logging.WARN, f' * No matcher results for {file}')
            df = pd.DataFrame()
        return df

    def sample_dataset(self) -> List[Dict[str, Any]]:
        lines = []

        scores = {}
        for system in self.systems:
            scores[system] = self.f_normalize(self.read_scores_file(self.params['scores'][system]))

        docs = dict_from_paths(self.params['docs'])
        doc_ids = set(pd.concat([df for system, df in scores.items() if system in self.systems]).document_id)
        for doc in doc_ids:
            head, tail = os.path.split(doc)
            tail = tail.split('.')[0]

            with open(docs[tail]) as fp:
                tokenized = self.tokenizer.tokenize(fp.read())
                tokenized = [w.orth_ for w in tokenized]

                lines.append({
                    'document_id': tail,
                    'base_path': head,
                    'text': ' '.join(tokenized),
                    'scores': self.get_scores(scores, doc)
                })

        return lines
