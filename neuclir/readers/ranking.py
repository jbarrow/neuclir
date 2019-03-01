from typing import Iterator, List, Dict, Tuple
from .utils import tokenize
from io import StringIO

from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField, ArrayField, ListField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

import numpy as np
import csv
import json


@DatasetReader.register('ranking_dataset_reader')
class RankingDatasetReader(DatasetReader):
    def __init__(self, scores: bool = True, query_token_indexers: Dict[str, TokenIndexer] = None,
                 doc_token_indexers: Dict[str, TokenIndexer] = None, lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self.scores = scores
        self.q_token_indexers = query_token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.d_token_indexers = doc_token_indexers or {'tokens': SingleIdTokenIndexer()}

    def line_to_instance(self, query: List[Token],
                         *docs: List[Tuple[List[Token], float, int]]) -> Instance:
        query_field = TextField(query, self.q_token_indexers)
        doc_fields = [TextField(doc[0], self.d_token_indexers) for doc in docs]

        fields = {
            'query': query_field,
            'docs': ListField(doc_fields),
        }

        if self.scores:
            lex_fields = [ArrayField(np.array([doc[1]])) for doc in docs]
            fields['scores'] = ListField(lex_fields)

        label_fields = [ArrayField(np.array([doc[2]])) for doc in docs]
        fields['labels'] = ListField(label_fields)

        # used to compute full AQWV and MAP scores from partial data
        relevant_ignored_field = ArrayField(np.array([0]))
        fields['relevant_ignored'] = relevant_ignored_field
        irrelevant_ignored_field = ArrayField(np.array([0]))
        fields['irrelevant_ignored'] = irrelevant_ignored_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as fp:
            for line in fp:
                line = json.loads(line)
                instance = self.line_to_instance(
                    tokenize(line['query']),
                    *[(tokenize(d['text']), float(d['scores'][0]['score']), int(d['relevant'])) for d in line['docs']],
                )
                yield instance
