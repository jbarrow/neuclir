import torch

from .scorer import Scorer
from typing import Dict, Optional, List, Any

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, masked_softmax, weighted_sum
from allennlp.training.metrics import CategoricalAccuracy


@Scorer.register("decomposable_attention_sentence_scorer")
class DecomposableAttentionSentenceScorer(Scorer):
    """
    This ``Model`` implements the Decomposable Attention model described in `"A Decomposable
    Attention Model for Natural Language Inference"
    <https://www.semanticscholar.org/paper/A-Decomposable-Attention-Model-for-Natural-Languag-Parikh-T%C3%A4ckstr%C3%B6m/07a9478e87a8304fc3267fa16e83e9f3bbd98b27>`_
    by Parikh et al., 2016, with some optional enhancements before the decomposable attention
    actually happens.  Parikh's original model allowed for computing an "intra-sentence" attention
    before doing the decomposable entailment step.  We generalize this to any
    :class:`Seq2SeqEncoder` that can be applied to the query and/or the document before
    computing entailment.

    The basic outline of this model is to get an embedded representation of each word in the
    query and document, align words between the two, compare the aligned phrases, and make a
    final entailment decision based on this aggregated comparison.  Each step in this process uses
    a feedforward network to modify the representation.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``query`` and ``document`` ``TextFields`` we get as input to the
        model.
    attend_feedforward : ``FeedForward``
        This feedforward network is applied to the encoded sentence representations before the
        similarity matrix is computed between words in the query and words in the document.
    similarity_function : ``SimilarityFunction``
        This is the similarity function used when computing the similarity matrix between words in
        the query and words in the document.
    compare_feedforward : ``FeedForward``
        This feedforward network is applied to the aligned query and document representations,
        individually.
    aggregate_feedforward : ``FeedForward``
        This final feedforward network is applied to the concatenated, summed result of the
        ``compare_feedforward`` network, and its output is used as the entailment class logits.
    query_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        After embedding the query, we can optionally apply an encoder.  If this is ``None``, we
        will do nothing.
    document_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        After embedding the document, we can optionally apply an encoder.  If this is ``None``,
        we will use the ``query_encoder`` for the encoding (doing nothing if ``query_encoder``
        is also ``None``).
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    """
    def __init__(self,
                 attend_feedforward: FeedForward,
                 similarity_function: SimilarityFunction,
                 compare_feedforward: FeedForward,
                 aggregate_feedforward: FeedForward,
                 document_encoder: Seq2VecEncoder,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(DecomposableAttentionSentenceScorer, self).__init__()

        self._attend_feedforward = TimeDistributed(attend_feedforward)
        self._matrix_attention = LegacyMatrixAttention(similarity_function)
        self._compare_feedforward = TimeDistributed(compare_feedforward)
        self._aggregate_feedforward = aggregate_feedforward
        self._document_encoder = document_encoder

        d_dim = self._document_encoder.get_output_dim()

        self._scorer = FeedForward(
            input_dim=2*d_dim, num_layers=1,
            hidden_dims=1, activations=lambda x: x, dropout=0.)

        initializer(self)


    def forward(self,  # type: ignore
                embedded_query: torch.Tensor,
                embedded_document: torch.Tensor,
                query_mask: torch.Tensor = None,
                sentence_mask: torch.Tensor = None,
                document_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ

        # (batch_size, num_sentences, sentence_length, embedding_dim)
        batch_size, num_sentences, sentence_length, embedding_dim = embedded_document.shape

        embedded_document = embedded_document.view(batch_size*num_sentences, sentence_length, embedding_dim)
        sentence_mask = sentence_mask.view(batch_size*num_sentences, sentence_length)

        # (batch_size, num_sentences, query_length, embedding_dim)
        embedded_query = embedded_query.unsqueeze(1).repeat(1, num_sentences, 1, 1)
        # (batch_size*num_sentences, query_length, embedding_dim)
        embedded_query = embedded_query.view(batch_size*num_sentences, -1, embedding_dim)
        # (batch_size, num_sentences, query_length)
        query_mask = query_mask.unsqueeze(1).repeat(1, num_sentences, 1)
        # (batch_size*num_sentences, query_length)
        query_mask = query_mask.view(batch_size*num_sentences, -1)

        projected_query = self._attend_feedforward(embedded_query)
        projected_document = self._attend_feedforward(embedded_document)
        # Shape: (batch_size, query_length, document_length)
        similarity_matrix = self._matrix_attention(projected_query, projected_document)

        # Shape: (batch_size, query_length, document_length)
        q2d_attention = masked_softmax(similarity_matrix, sentence_mask)
        # Shape: (batch_size, query_length, embedding_dim)
        attended_document = weighted_sum(embedded_document, q2d_attention)

        # Shape: (batch_size, document_length, query_length)
        d2q_attention = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), query_mask)
        # Shape: (batch_size, document_length, embedding_dim)
        attended_query = weighted_sum(embedded_query, d2q_attention)

        query_compare_input = torch.cat([embedded_query, attended_document], dim=-1)
        document_compare_input = torch.cat([embedded_document, attended_query], dim=-1)

        compared_query = self._compare_feedforward(query_compare_input)
        compared_query = compared_query * query_mask.unsqueeze(-1).float()
        # Shape: (batch_size, compare_dim)
        compared_query = compared_query.sum(dim=1)

        compared_document = self._compare_feedforward(document_compare_input)
        compared_document = compared_document * sentence_mask.unsqueeze(-1).float()
        # Shape: (batch_size, compare_dim)
        compared_document = compared_document.sum(dim=1)

        aggregate_input = torch.cat([compared_query, compared_document], dim=-1)
        transformed_input = self._aggregate_feedforward(aggregate_input).squeeze(1)
        transformed_input = transformed_input.view(batch_size, num_sentences, embedding_dim)

        doc_representation = self._document_encoder(transformed_input)

        compared_query = compared_query.view(batch_size, num_sentences, -1).mean(dim=1)

        reprs = [compared_query - doc_representation, compared_query * doc_representation]

        encoded = torch.cat(reprs, dim=1)

        label_logits = self._scorer(encoded)

        return label_logits
