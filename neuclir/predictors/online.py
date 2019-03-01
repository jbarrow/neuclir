from allennlp.predictors.predictor import Predictor
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from typing import List

from ..models import *
from ..readers.utils import tokenize

import torch.nn.functional as F
import torch
import json

epsilon = 1e-6


@Predictor.register('online_predictor')
class OnlinePredictor(Predictor):
    def predict_json(self, query: str, docs: List[JsonDict]) -> JsonDict:
        # get the base_paths
        base_paths = [doc['base_path'] for doc in docs]
        # get all the doc_ids for the output
        doc_ids = [doc['document_id'] for doc in docs]
        # generate a training instance for the query
        instance = self._json_to_instance(query, docs)
        # get the scores for test
        predictions =  F.log_softmax(torch.tensor(self.predict_instance(instance)['logits']), dim=0)
        predictions = [p.item() - epsilon for p in predictions]
        # get the sorted list of documents out
        return sorted(
            zip(base_paths, doc_ids, predictions),
            #zip(base_paths, doc_ids, scores),
            key=lambda x: x[-1], reverse=True)

    def _json_to_instance(self, query: str, docs: JsonDict) -> Instance:
        return self._dataset_reader.line_to_instance(
            tokenize(query),
            [(tokenize(d['text']), float(d['scores'][0]['score']), 0) for d in docs]
        )
