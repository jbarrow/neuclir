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

@Model.register('letor_sentence_training')
class LeToRSentenceWrapper(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 doc_field_embedder: TextFieldEmbedder,
                 scorer: Scorer,
                 validation_metrics: Dict[str, Metric],
                 temperature: float = 2.0,
                 alpha: float = 0.8,
                 ranking_loss: bool = False,
                 query_field_embedder: Optional[TextFieldEmbedder] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 idf_embedder: Optional[TextFieldEmbedder] = None,
                 dropout: Optional[float] = 0.) -> None:
        super(LeToRSentenceWrapper, self).__init__(vocab, regularizer)

        self.query_field_embedder = query_field_embedder
        if self.query_field_embedder is None:
            self.query_field_embedder = doc_field_embedder

        self.doc_field_embedder = doc_field_embedder
        self.idf_embedder = idf_embedder

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

        self.classification_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

        # self.ranking_loss = ranking_loss
        # if self.ranking_loss:
        #     self.loss = nn.MarginRankingLoss(margin=1.0)
        # else:
        #     self.loss = nn.CrossEntropyLoss()
        initializer(self)

    def distillation_loss(self, y, teacher_scores, labels=None):
        #teacher_scores = torch.exp(teacher_scores)
        p = F.log_softmax(y / self.temperature, dim=1)
        q = F.softmax(teacher_scores / self.temperature, dim=1)

        # if not self.training:
        #     print('TARGET:', q)
        #     print('INPUTS:', y)
        #     print('LABELS:', labels)

        l_kl = self.kl_loss(p, q) * self.temperature * self.temperature
        l_ce = 0.
        if labels is not None:
           l_ce = self.classification_loss(y, labels)

        l_total = l_kl * self.kd_alpha + l_ce * (1. - self.kd_alpha)
        return l_total

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
        # (batch_size, num_docs, num_sentences, sentence_length)
        ds_mask = get_text_field_mask(docs, num_wrapping_dims=1)
        ss_mask = get_text_field_mask(docs, num_wrapping_dims=2)
        # (batch_size, num_docs, num_sentences, sentence_length, embedding_dim)
        ds_embedded = self.doc_field_embedder(docs)
        batch_size, num_docs, num_sentences, sentence_length, embedding_dim = ds_embedded.shape
        # (batch_size * num_docs, num_sentences, sentence_length, embedding_dim)
        ds_embedded = ds_embedded.view(batch_size*num_docs, num_sentences, sentence_length, embedding_dim)
        ds_mask = ds_mask.view(batch_size*num_docs, num_sentences)
        ss_mask = ss_mask.view(batch_size*num_docs, num_sentences, sentence_length)

        # (batch_size, query_length)
        qs_mask = get_text_field_mask(query)
        _, query_length = qs_mask.shape
        qs_mask = qs_mask.unsqueeze(1).repeat(1, num_docs, 1).view(batch_size*num_docs, -1)
        # (batch_size, query_length, embedding_dim)
        qs_embedded = self.query_field_embedder(query)
        # (batch_size, num_docs, query_length, embedding_dim)
        qs_embedded = qs_embedded.unsqueeze(1).repeat(1, num_docs, 1, 1)
        # (batch_size * num_docs, query_length, embedding_dim)
        qs_embedded = qs_embedded.view(batch_size*num_docs, query_length, embedding_dim)

        logits = self.scorer(qs_embedded, ds_embedded, qs_mask, ss_mask, ds_mask).view(batch_size, num_docs)

        output_dict = {'logits': logits}

        if scores is not None:
            output_dict['loss'] = self.distillation_loss(logits, scores, labels)

        if labels is not None:
            sfl = F.log_softmax(logits,dim=1)
            output_dict['accuracy'] = self.metrics['accuracy'](sfl, labels)

        return output_dict
