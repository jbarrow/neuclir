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

from typing import Optional, Dict, Any, List
from ..metrics import AQWV
from .scorers import Scorer

from allennlp.modules.attention.cosine_attention import CosineAttention

@Model.register('letor_training_score')
class LeToRWrapper(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 query_field_embedder: TextFieldEmbedder,
                 doc_field_embedder: TextFieldEmbedder,
                 scorer: Scorer,
                 validation_metrics: Dict[str, Metric],
                 temperature: float = 15.0,
                 alpha: float = 0.8,
                 ranking_loss: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 idf_embedder: Optional[TextFieldEmbedder] = None,
                 dropout: float = 0.) -> None:
        super(LeToRWrapper, self).__init__(vocab, regularizer)

        self.embedder = doc_field_embedder
        self.idf_embedder = idf_embedder
        self.final_scorer = FeedForward(2, 1, 1, lambda x: x)

        self.scorer = scorer

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

        self.temperature = temperature
        self.kd_alpha = alpha

        # self.ranking_loss = ranking_loss
        # if self.ranking_loss:
        #self.loss = nn.MarginRankingLoss(margin=1.0)
        # else:
        self.loss = nn.CrossEntropyLoss()
        initializer(self)
    #
    # def distillation_loss(self, y, teacher_scores, labels=None):
    #     p = F.log_softmax(y / self.temperature, dim=1)
    #     q = F.softmax(teacher_scores / self.temperature, dim=1)
    #     l_kl = F.kl_div(p, q, size_average=False) * (self.temperature ** 2) / \
    #            y.shape[0]
    #
    #     l_ce = 0.
    #     if labels is not None:
    #         l_ce = F.cross_entropy(y, labels)
    #
    #     return l_kl * self.kd_alpha + l_ce * (1. - self.kd_alpha)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            metric_name: metric.get_metric(reset)
                for metric_name, metric in self.metrics.items() }

    def forward(self,
                query: Dict[str, torch.LongTensor],
                docs: Dict[str, torch.LongTensor],
                dataset: List[str] = [],
                labels: Optional[Dict[str, torch.LongTensor]] = None,
                scores: Optional[Dict[str, torch.Tensor]] = None,
                relevant_ignored: Optional[torch.Tensor] = None,
                irrelevant_ignored: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # label masks
        ls_mask = get_text_field_mask(docs)
        # (batch_size, num_docs, doc_length)
        ds_mask = get_text_field_mask(docs, num_wrapping_dims=1)
        # (batch_size, num_docs, doc_length, embedding_dim)
        ds_embedded = self.embedder(docs)
        # (batch_size, num_docs, doc_length, transform_dim)
        batch_size, num_docs, doc_length, embedding_dim = ds_embedded.shape
        # (batch_size * num_docs, doc_length, transform_dim)
        ds_embedded = ds_embedded.view(batch_size*num_docs, doc_length, embedding_dim)
        ds_mask = ds_mask.view(batch_size*num_docs, doc_length)

        if self.idf_embedder is not None:
            ds_idfs = self.idf_embedder(docs)
            ds_idfs = ds_idfs.view(batch_size*num_docs, doc_length, 1).repeat(1, 1, embedding_dim)
            ds_embedded = ds_embedded * ds_idfs

        # (batch_size, query_length)
        qs_mask = get_text_field_mask(query)
        _, query_length = qs_mask.shape
        qs_mask = qs_mask.unsqueeze(1).repeat(1, num_docs, 1).view(batch_size*num_docs, -1)
        # (batch_size, query_length, embedding_dim)
        qs_embedded = self.embedder(query)
        # (batch_size, num_docs, query_length, embedding_dim)
        qs_embedded = qs_embedded.unsqueeze(1).repeat(1, num_docs, 1, 1)
        # (batch_size, num_docs, query_length, embedding_dim)
        qs_embedded = qs_embedded.view(batch_size*num_docs, query_length, embedding_dim)

        if self.idf_embedder is not None:
            qs_idfs = self.idf_embedder(query).unsqueeze(1).repeat(1, num_docs, 1, 1)
            qs_idfs = qs_idfs.view(batch_size*num_docs, query_length, 1).repeat(1, 1, embedding_dim)
            qs_embedded = qs_embedded * qs_idfs

        logits = self.scorer(qs_embedded, ds_embedded, qs_mask, ds_mask)
        #logits = F.log_softmax(logits,dim=1)

        scores = torch.exp(scores / self.temperature).view(batch_size*num_docs, -1)
        logits = torch.cat([logits, scores], dim=1)

        logits = self.final_scorer(logits).view(batch_size, num_docs)

        output_dict = {'logits': logits}

        if labels is not None:
            # filter out to only the metrics we care about
            # if self.training:
            #     if self.ranking_loss:
            #         loss = self.loss(logits[:, 0], logits[:, 1], labels.float()*-2.+1.)
            #     else:
            #         loss = self.loss(logits, labels.squeeze(-1).long())

            #self.metrics['accuracy'](logits, labels.squeeze(-1))
            # else:
            #    # at validation time, we can't compute a proper loss
            #    loss = torch.Tensor([0.])
            #    for metric in self.training_metrics[False]:
            #        self.metrics[metric](logits, labels.squeeze(-1).long(), ls_mask, relevant_ignored, irrelevant_ignored)
            #output_dict['loss'] = self.loss(logits[:, 0], logits[:, 1], labels.float()*2.+1.)
            output_dict['loss'] = self.loss(logits, labels)

        if labels is not None:
            sfl = F.log_softmax(logits,dim=1)
            #print(sfl)
            output_dict['accuracy'] = self.metrics['accuracy'](sfl, labels)

        return output_dict
