from typing import Iterator, List, Dict, Tuple, Optional
from neuclir.readers.utils import tokenize

from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField, ArrayField, ListField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.fields import MetadataField

import numpy as np
import json


@DatasetReader.register('distillation_reader')
class DistillationDatasetReader(DatasetReader):
    def __init__(self,
                 dataset_name_field: str = 'dataset',
                 query_token_indexers: Dict[str, TokenIndexer] = None,
                 doc_token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self.dataset_name_field = dataset_name_field
        self.q_token_indexers = query_token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.d_token_indexers = doc_token_indexers or {'tokens': SingleIdTokenIndexer()}

    def line_to_instance(self, query: List[Token], docs: List[List[Token]],
                         relevant_ix: int = None, scores: List[float] = None,
                         dataset: Optional[str] = None) -> Instance:
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
            label_field = LabelField(int(relevant_ix), skip_indexing=True)
            fields['labels'] = label_field

        if dataset is not None:
            fields[self.dataset_name_field] = MetadataField(dataset)

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as fp:
            for line in fp:
                line = json.loads(line)
                dataset = None
                if self.dataset_name_field in line:
                    dataset = line[self.dataset_name_field]
                docs = line.get('docs', line.get('documents'))
                instance = self.line_to_instance(
                    tokenize(line['query']),
                    [tokenize(d.replace('</s>', '')) for d in docs],
                    scores = np.array(line['scores']),
                    relevant_ix=line.get('relevant_ix', None),
                    dataset=dataset
                )
                yield instance
