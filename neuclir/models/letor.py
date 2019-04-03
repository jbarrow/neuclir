import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.feedforward import FeedForward
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.initializers import InitializerApplicator
from allennlp.nn.regularizers import RegularizerApplicator
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.modules.seq2vec_encoders.boe_encoder import BagOfEmbeddingsEncoder
from allennlp.training.metrics.metric import Metric

from typing import Optional, Dict, Any
from ..metrics import AQWV
from .scorers import Scorer

from allennlp.modules.attention.cosine_attention import CosineAttention

@Model.register('letor_training')
class LeToRWrapper(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 query_field_embedder: TextFieldEmbedder,
                 doc_field_embedder: TextFieldEmbedder,
                 scorer: Scorer,
                 total_scorer: FeedForward,
                 validation_metrics: Dict[str, Metric],
                 ranking_loss: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 idf_embedder: Optional[TextFieldEmbedder] = None,
                 dropout: float = 0.) -> None:
        super(LeToRWrapper, self).__init__(vocab, regularizer)

        self.query_field_embedder = query_field_embedder
        self.doc_field_embedder = doc_field_embedder
        self.idf_embedder = idf_embedder

        self.scorer = scorer
        self.total_scorer = total_scorer

        self.initializer = initializer
        self.regularizer = regularizer

        self.metrics = copy.deepcopy(validation_metrics)
        self.metrics.update({
            'accuracy': CategoricalAccuracy()
        })

        self.training_metrics = {
            True: ['accuracy'],
            False: validation_metrics.keys()
        }

        self.ranking_loss = ranking_loss
        if self.ranking_loss:
            self.loss = nn.MarginRankingLoss(margin=1.0)
        else:
            self.loss = nn.CrossEntropyLoss()
        initializer(self)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            metric_name: metric.get_metric(reset)
                for metric_name, metric in self.metrics.items() }

    def forward(self,
                query: Dict[str, torch.LongTensor],
                docs: Dict[str, torch.LongTensor],
                labels: Optional[Dict[str, torch.LongTensor]] = None,
                scores: Optional[Dict[str, torch.Tensor]] = None,
                relevant_ignored: Optional[torch.Tensor] = None,
                irrelevant_ignored: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        # label masks
        ls_mask = get_text_field_mask(docs)
        # (batch_size, num_docs, doc_length)
        ds_mask = get_text_field_mask(docs, num_wrapping_dims=1)
        # (batch_size, num_docs, doc_length, embedding_dim)
        ds_embedded = self.doc_field_embedder(docs)
        # (batch_size, num_docs, doc_length, transform_dim)
        batch_size, num_docs, doc_length, embedding_dim = ds_embedded.shape
        # (batch_size * num_docs, doc_length, transform_dim)
        ds_transformed = ds_embedded.view(batch_size*num_docs, doc_length, embedding_dim)
        ds_mask = ds_mask.view(batch_size*num_docs, doc_length)

        # (batch_size, query_length)
        qs_mask = get_text_field_mask(query)
        _, query_length = qs_mask.shape
        qs_mask = qs_mask.unsqueeze(1).repeat(1, num_docs, 1).view(batch_size*num_docs, -1)
        # (batch_size, query_length, embedding_dim)
        qs_embedded = self.query_field_embedder(query)
        # (batch_size, num_docs, query_length, embedding_dim)
        qs_embedded = qs_embedded.unsqueeze(1).repeat(1, num_docs, 1, 1)
        # (batch_size, num_docs, query_length, embedding_dim)
        qs_embedded = qs_embedded.view(batch_size*num_docs, query_length, embedding_dim)

        semantic_scores = self.scorer(qs_embedded, ds_embedded, qs_mask, ds_mask)

        if scores is not None:
            # (batch_size, num_docs, 2)
            semantic_scores = torch.cat([semantic_scores, scores], dim=2)

        # (batch_size, num_docs)
        logits = self.total_scorer(semantic_scores)
        logits = logits.squeeze(2)

        output_dict = {'logits': logits}

        if labels is not None:
            # filter out to only the metrics we care about
            if self.training:
                if self.ranking_loss:
                    loss = self.loss(logits[:, 0], logits[:, 1], labels.float()*-2.+1.)
                else:
                    loss = self.loss(logits, labels.squeeze(-1).long())
                self.metrics['accuracy'](logits, labels.squeeze(-1))
            else:
               # at validation time, we can't compute a proper loss
               loss = torch.Tensor([0.])
               for metric in self.training_metrics[False]:
                   self.metrics[metric](logits, labels.squeeze(-1).long(), ls_mask, relevant_ignored, irrelevant_ignored)
            output_dict["loss"] = loss

        return output_dict
