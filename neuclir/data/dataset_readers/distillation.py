from typing import Iterator, List, Dict, Tuple
from neuclir.readers.utils import tokenize

from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField, ArrayField, ListField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

import numpy as np
import json


@DatasetReader.register('distillation-reader')
class DistillationDatasetReader(DatasetReader):
    def __init__(self,
                 query_token_indexers: Dict[str, TokenIndexer] = None,
                 doc_token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self.q_token_indexers = query_token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.d_token_indexers = doc_token_indexers or {'tokens': SingleIdTokenIndexer()}

    def line_to_instance(self, query: List[Token], docs: List[List[Token]],
                         relevant_ix: int = None, scores: List[float] = None) -> Instance:
        query_field = TextField(query, self.q_token_indexers)
        doc_fields = [TextField(doc, self.d_token_indexers) for doc in docs]

        fields = {
            'query': query_field,
            'docs': ListField(doc_fields)
        }

        if scores is not None:
            scores_field = ArrayField(scores)
            fields['scores'] = scores_field

        if relevant_ix is not None:
            label_field = LabelField(relevant_ix)
            fields['labels'] = label_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as fp:
            for line in fp:
                line = json.loads(line)
                instance = self.line_to_instance(
                    tokenize(line['query']),
                    [tokenize(d['text']) for d in line['docs']],
                    scores = np.array(line['distribution'])
                )
                yield instance
