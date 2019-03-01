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


@Predictor.register('query_predictor')
class QueryPredictor(Predictor):
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        doc_ids = [d['document_id'] for d in inputs['docs']]
        # run the model forward
        instance = self._json_to_instance(inputs)
        # print(list(F.log_softmax(torch.tensor(self.predict_instance(instance)['logits']),dim=0)))
        predictions =  F.log_softmax(torch.tensor(self.predict_instance(instance)['logits']), dim=0)
        predictions = [p.item() - epsilon for p in predictions]
        #predictions = [d['scores'][0]['score'] for d in inputs['docs']]
        predictions = sorted(zip(doc_ids, predictions), key=lambda x: x[-1], reverse=True)#[:2]
        # get the sorted list of documents out
        return {
            'query_id': inputs['query_id'],
            'scores': predictions
        }

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.line_to_instance(
            tokenize(json_dict['query']),
            [(tokenize(d['text']), float(d['scores'][0]['score']), 0) for d in json_dict['docs']]
        )

    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        """
        Rather than predicting batches, I just take the easy way out and compute
        predictions serially. Typically I can't fit more than 2 queries in memory
        at once for prediction anyways, so I'm not losing too much time here.
        """
        outputs = []
        for query in inputs:
            outputs.append(self.predict_json(query))
        return outputs
