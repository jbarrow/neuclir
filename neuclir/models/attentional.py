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
from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from allennlp.modules.seq2vec_encoders.boe_encoder import BagOfEmbeddingsEncoder

from typing import Optional, Dict, Any
from ..metrics import AQWV


@Model.register('attentional_training')
class LeToRWrapper(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 query_field_embedder: TextFieldEmbedder,
                 doc_field_embedder: TextFieldEmbedder,
                 doc_transformer: FeedForward,
                 query_transformer: FeedForward,
                 scorer: FeedForward,
                 total_scorer: FeedForward,
                 doc_encoder: Seq2VecEncoder = BagOfEmbeddingsEncoder(50),
                 query_encoder: Seq2VecEncoder = BagOfEmbeddingsEncoder(50),
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 aqwv_corrections: Optional[str] = None,
                 predicting: Optional[bool] = False) -> None:
        super(LeToRWrapper, self).__init__(vocab, regularizer)

        self.query_field_embedder = query_field_embedder
        self.doc_field_embedder = doc_field_embedder
        self.document_transformer = doc_transformer
        self.query_transformer = query_transformer

        self.scorer = scorer
        self.total_scorer = total_scorer
        #self.scorer_norm = nn.BatchNorm1d(1)
        self.doc_encoder = doc_encoder
        self.doc_norm = nn.BatchNorm1d(50)
        self.score_norm = nn.BatchNorm1d(3)
        self.query_encoder = query_encoder
        #self.query_norm = nn.BatchNorm1d(100)
        self.initializer = initializer
        self.regularizer = regularizer
        #self.score_dropout = nn.Dropout(0.05)
        self.emb_dropout = nn.Dropout(0.2)

        self.encoder = CnnEncoder(embedding_dim=1, num_filters=20, output_dim=1)
        #self.doc_encoder = CnnEncoder(embedding_dim=50, num_filters=20, output_dim=50)

        if not predicting:
            self.metrics = {
                'accuracy': CategoricalAccuracy(),
                'aqwv_2': AQWV(corrections_file=aqwv_corrections, cutoff=2, version='program'),
                #'aqwv_3': AQWV(corrections_file=aqwv_corrections, cutoff=3, version='program')
            }

            self.training_metrics = {
                True: ['accuracy'],
                #False: ['accuracy']
                False: ['aqwv_2']#, 'aqwv_3']
            }
        else:
            self.metrics, self.training_metrics = {}, { True: [], False: [] }

        #self.loss = nn.MarginRankingLoss(margin=0.5)
        self.loss = nn.CrossEntropyLoss()
        initializer(self)

    def get_metrics(self, reset: bool = True) -> Dict[str, float]:
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
        batch_size, num_docs, doc_length, transform_dim = ds_transformed.shape
        # (batch_size * num_docs, doc_length, transform_dim)
        ds_transformed_prime = ds_transformed.view(batch_size*num_docs, doc_length, transform_dim)
        ds_mask = ds_mask.view(batch_size*num_docs, doc_length)
        # (batch_size * num_docs, transform_dim)
        ds_encoded = self.doc_encoder(ds_transformed_prime, ds_mask)
        ds_encoded = self.doc_norm(ds_encoded)
        # (batch_size, num_docs, transform_dim)
        ds_encoded = ds_encoded.view(batch_size, num_docs, transform_dim)

        # (batch_size, query_length)
        qs_mask = get_text_field_mask(query)
        # (batch_size, query_length, embedding_dim)
        qs_embedded = self.query_field_embedder(query)
        # (batch_size, query_length, transform_dim)
        #qs_transformed = self.query_transformer(qs_embedded)
        #qs_transformed = self.query_norm(qs_encoded)
        qs_transformed = qs_embedded
        # (batch_size, transform_dim)
        qs_encoded = self.query_encoder(qs_transformed, qs_mask)
        # (batch_size, 1, 1, transform_dim)
        qs_encoded_prime = qs_encoded.unsqueeze(1).repeat(1, num_docs, 1)
        qs_encoded = qs_encoded.unsqueeze(1).unsqueeze(1)

        similarities = F.cosine_similarity(ds_transformed, qs_encoded, dim=3)
        similarities = similarities.view(batch_size*num_docs, doc_length, 1)

        lexical_scores = self.encoder(similarities, ds_mask).view(batch_size, num_docs, 1)
        #print(semantic_scores.shape, scores.shape)

        # (batch_size, num_docs, transform_dim * 2 + 1)
        #qd = torch.cat([qs_encoded - ds_encoded, qs_encoded * ds_encoded, F.cosine_similarity(ds_encoded, qs_encoded, dim=2).unsqueeze(2)], dim=2)
        qd = torch.cat([qs_encoded_prime - ds_encoded, qs_encoded_prime * ds_encoded], dim=2)
        qd = self.emb_dropout(qd)

        semantic_scores = self.scorer(qd)
        #semantic_scores = self.score_dropout(semantic_scores)

        if scores is not None:
            # (batch_size, num_docs, transform_dim * 2 + 1)
            semantic_scores = torch.cat([semantic_scores, lexical_scores, scores], dim=2)

        semantic_scores = semantic_scores.view(batch_size * num_docs, -1)
        semantic_scores = self.score_norm(semantic_scores)
        semantic_scores = semantic_scores.view(batch_size, num_docs, -1)

        # dropout the scores for robustness

        # (batch_size, num_docs)
        logits = self.total_scorer(semantic_scores)
        #logits = self.scorer_norm(logits)
        logits = logits.squeeze(2)
        # logits = self.scorer(scores).squeeze(2)
        # scores = scores.squeeze(2)
        #print(scores.shape, logits.shape)

        output_dict = {'logits': logits}

        if labels is not None:
            # filter out to only the metrics we care about
            if self.training:
                #print(labels.float().shape, scores.shape)
                #loss = self.loss(logits[:, 0], logits[:, 1], labels.float()*-2.+1.)
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
