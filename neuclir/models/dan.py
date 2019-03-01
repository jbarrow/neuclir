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

from allennlp.modules.attention.cosine_attention import CosineAttention

@Model.register('full_model')
class FullModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 query_field_embedder: TextFieldEmbedder,
                 doc_field_embedder: TextFieldEmbedder,
                 #doc_transformer: FeedForward,
                 #query_transformer: FeedForward,
                 scorer: FeedForward,
                 total_scorer: FeedForward,
                 validation_metrics: Dict[str, Metric],
                 use_attention: bool = True,
                 use_batch_norm: bool = True,
                 ranking_loss: bool = False,
                 doc_encoder: Seq2VecEncoder = BagOfEmbeddingsEncoder(50),
                 query_encoder: Seq2VecEncoder = BagOfEmbeddingsEncoder(50),
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 idf_embedder: Optional[TextFieldEmbedder] = None,
                 #aqwv_corrections: Optional[str] = None,
                 #aqwv_test_corrections: Optional[str] = None,
                 predicting: Optional[bool] = False,
                 dropout: float = 0.) -> None:
        super(LeToRWrapper, self).__init__(vocab, regularizer)

        self.query_field_embedder = query_field_embedder
        self.doc_field_embedder = doc_field_embedder
        self.idf_embedder = idf_embedder
        #self.document_transformer = doc_transformer
        #self.query_transformer = query_transformer

        self.scorer = scorer
        self.total_scorer = total_scorer
        self.doc_encoder = doc_encoder
        self.query_encoder = query_encoder

        self.initializer = initializer
        self.regularizer = regularizer
        self.dropout = nn.Dropout(dropout)

        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            self.qd_norm = nn.BatchNorm1d(self.doc_encoder.get_output_dim()*2)
            self.score_norm = nn.BatchNorm1d(2)

        self.use_attention = use_attention
        if self.use_attention:
            self.attn = CosineAttention()

        if not predicting:
            self.metrics = copy.deepcopy(validation_metrics)
            self.metrics.update({
                'accuracy': CategoricalAccuracy()
            })

            self.training_metrics = {
                True: ['accuracy'],
                False: validation_metrics.keys()
            }
        else:
            self.metrics, self.training_metrics = {}, { True: [], False: [] }

        self.ranking_loss = ranking_loss
        if self.ranking_loss:
            self.loss = nn.MarginRankingLoss(margin=0.5)
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

        _, num_docs, _ = docs['tokens'].shape

        # (batch_size, query_length)
        qs_mask = get_text_field_mask(query)
        # (batch_size, query_length, embedding_dim)
        qs_embedded = self.query_field_embedder(query)
        #qs_idfs = self.idf_embedder(query)
        # (batch_size, query_length, transform_dim)
        #qs_transformed = self.query_transformer(qs_embedded)
        #qs_transformed = self.query_norm(qs_encoded)
        qs_transformed = qs_embedded# * qs_idfs
        # (batch_size, transform_dim)
        qs_encoded = self.query_encoder(qs_transformed, qs_mask)
        # (batch_size, num_docs, transform_dim)
        qs_encoded = qs_encoded.unsqueeze(1).repeat(1, num_docs, 1)

        # label masks
        ls_mask = get_text_field_mask(docs)
        # (batch_size, num_docs, doc_length)
        ds_mask = get_text_field_mask(docs, num_wrapping_dims=1)
        # (batch_size, num_docs, doc_length, embedding_dim)
        ds_embedded = self.doc_field_embedder(docs)
        # (batch_size, num_docs, doc_length, transform_dim)
        #ds_transformed = self.document_transformer(ds_embedded)
        #ds_transformed = self.doc_norm(ds_transformed)
        ds_transformed = ds_embedded
        if self.idf_embedder is not None:
            # (batch_size, num_docs, doc_length, 1)
            ds_idfs = self.idf_embedder(docs)
            ds_transformed = ds_transformed * ds_idfs

        batch_size, num_docs, doc_length, transform_dim = ds_transformed.shape
        # (batch_size * num_docs, doc_length, transform_dim)
        ds_transformed = ds_transformed.view(batch_size*num_docs, doc_length, transform_dim)
        ds_mask = ds_mask.view(batch_size*num_docs, doc_length)

        if self.use_attention:
            qs_encoded = qs_encoded.view(batch_size*num_docs, -1)
            attn = self.attn(qs_encoded, ds_transformed, ds_mask).unsqueeze(2)
            qs_encoded = qs_encoded.view(batch_size, num_docs, -1)
            ds_transformed = ds_transformed * attn

        # (batch_size * num_docs, transform_dim)
        ds_encoded = self.doc_encoder(ds_transformed, ds_mask)
        # (batch_size, num_docs, transform_dim)
        ds_encoded = ds_encoded.view(batch_size, num_docs, transform_dim)

        # (batch_size, num_docs, transform_dim * 2 + 1)
        #qd = torch.cat([qs_encoded - ds_encoded, qs_encoded * ds_encoded, F.cosine_similarity(ds_encoded, qs_encoded, dim=2).unsqueeze(2)], dim=2)
        qd = torch.cat([qs_encoded - ds_encoded, qs_encoded * ds_encoded], dim=2)
        #qd = torch.cat([qs_encoded - ds_encoded, qs_encoded * ds_encoded, scores], dim=2)

        if self.use_batch_norm:
            qd = qd.view(batch_size*num_docs, -1)
            qd = self.qd_norm(qd)
            qd = qd.view(batch_size, num_docs, -1)

        qd = self.dropout(qd)

        semantic_scores = self.scorer(qd)

        if scores is not None:
            # (batch_size, num_docs, 2)
            semantic_scores = torch.cat([semantic_scores, scores], dim=2)

            if self.use_batch_norm:
                semantic_scores = semantic_scores.view(batch_size*num_docs, -1)
                semantic_scores = self.score_norm(semantic_scores)
                semantic_scores = semantic_scores.view(batch_size, num_docs, -1)

        # (batch_size, num_docs)
        logits = self.total_scorer(semantic_scores)
        # logits = semantic_scores
        logits = logits.squeeze(2)
        # logits = self.scorer(scores).squeeze(2)
        # scores = scores.squeeze(2)
        #print(scores.shape, logits.shape)

        output_dict = {'logits': logits}

        if labels is not None:
            # filter out to only the metrics we care about
            if True or self.training:
                #print(labels.float().shape, scores.shape)
                if self.ranking_loss:
                    loss = self.loss(logits[:, 0], logits[:, 1], labels.float()*-2.+1.)
                else:
                    loss = self.loss(logits, labels.squeeze(-1).long())
                self.metrics['accuracy'](logits, labels.squeeze(-1))
            else:
               # at validation time, we can't compute a proper loss
               loss = torch.Tensor([0.])
               for metric in self.training_metrics[False]:
                   #print(relevant_ignored.squeeze(), irrelevant_ignored.squeeze(), ls_mask.sum(dim=1).squeeze())
                   self.metrics[metric](logits, labels.squeeze(-1).long(), ls_mask, relevant_ignored, irrelevant_ignored)
            # metrics = [value for name, value in self.metrics.items() if name in self.training_metrics[self.training]]
            # for metric in metrics:
            #     metric(logits, labels.squeeze(-1))
            output_dict["loss"] = loss

        return output_dict
