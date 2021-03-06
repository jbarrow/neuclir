import torch

from .scorer import Scorer
from typing import Dict, Optional, List, Any

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, masked_softmax, weighted_sum
from allennlp.training.metrics import CategoricalAccuracy


@Scorer.register("decomposable_attention_scorer")
class DecomposableAttentionScorer(Scorer):
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
                 query_encoder: Optional[Seq2SeqEncoder] = None,
                 document_encoder: Optional[Seq2SeqEncoder] = None,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(DecomposableAttentionScorer, self).__init__()

        self._attend_feedforward = TimeDistributed(attend_feedforward)
        self._matrix_attention = LegacyMatrixAttention(similarity_function)
        self._compare_feedforward = TimeDistributed(compare_feedforward)
        self._aggregate_feedforward = aggregate_feedforward
        self._query_encoder = query_encoder
        self._document_encoder = document_encoder or query_encoder

        initializer(self)


    def forward(self,  # type: ignore
                embedded_query: torch.Tensor,
                embedded_document: torch.Tensor,
                query_mask: torch.Tensor = None,
                document_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        if self._query_encoder:
            embedded_query = self._query_encoder(embedded_query, query_mask)
        if self._document_encoder:
            embedded_document = self._document_encoder(embedded_document, document_mask)

        T = self._temperature()

        projected_query = self._attend_feedforward(embedded_query)
        projected_document = self._attend_feedforward(embedded_document)
        # Shape: (batch_size, query_length, document_length)
        similarity_matrix = self._matrix_attention(projected_query, projected_document)

        # Shape: (batch_size, query_length, document_length)
        q2d_attention = masked_softmax(similarity_matrix / T, document_mask)
        print(q2d_attention[:, :, :5])
        print(q2d_attention.max(2)[0])
        # Shape: (batch_size, query_length, embedding_dim)
        attended_document = weighted_sum(embedded_document, q2d_attention)

        # Shape: (batch_size, document_length, query_length)
        d2q_attention = masked_softmax(similarity_matrix.transpose(1, 2).contiguous() / T, query_mask)
        # Shape: (batch_size, document_length, embedding_dim)
        attended_query = weighted_sum(embedded_query, d2q_attention)

        query_compare_input = torch.cat([embedded_query, attended_document], dim=-1)
        document_compare_input = torch.cat([embedded_document, attended_query], dim=-1)

        compared_query = self._compare_feedforward(query_compare_input)
        compared_query = compared_query * query_mask.unsqueeze(-1).float()
        # Shape: (batch_size, compare_dim)
        compared_query = compared_query.sum(dim=1)

        compared_document = self._compare_feedforward(document_compare_input)
        compared_document = compared_document * document_mask.unsqueeze(-1).float()
        # Shape: (batch_size, compare_dim)
        compared_document = compared_document.sum(dim=1)

        aggregate_input = torch.cat([compared_query, compared_document], dim=-1)
        label_logits = self._aggregate_feedforward(aggregate_input).squeeze(1)

        return label_logits
