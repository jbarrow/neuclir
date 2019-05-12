import torch
import torch.nn as nn

from .scorer import Scorer
from typing import Optional
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.modules.attention import Attention
from allennlp.nn import Activation

@Scorer.register("duet_scorer")
class DuetScorer(Scorer):
    """
    The ``Seq2VecScorer'' is a ``Scorer'' which computes a single vector
    representation for each query and document, and then computes the

    Parameters
    ----------
    doc_encoder: ``Seq2VecEncoder''
        This is the encoder for the document. Common options include the
        ``BagOfEmbeddingsEncoder'' or the ``CNNEncoder''.
    query_encoder: ``Seq2VecEncoder''
        This is the encoder for the query. Can be of the same type or a
        different type than that of the document encoder.
    use_encoded: ``bool'', optional (default = ``False'')
        Whether or not to use the encoded representations.
    scorer: ``FeedForward'', optional (default = ``None'')
        If no feedforward scorer is provided, we use a single-layer feedforward
        network that uses.
    attention: ``Attention'', optional (default = ``None'')
        If attention is provided, we use it to compute a query-specific document
        representation.
    """

    def __init__(self,
                 doc_encoder: Seq2VecEncoder,
                 query_encoder: Seq2VecEncoder,
                 use_encoded: bool = False,
                 scorer: Optional[FeedForward] = None,
                 attention: Optional[Attention] = None) -> None:

        super(Seq2VecScorer, self).__init__()

        self.doc_encoder = doc_encoder
        self.query_encoder = query_encoder
        self.use_encoded = use_encoded
        self.attention = attention
        # get the dimensions for the scorer and for sanity checking
        q_dim = self.query_encoder.get_output_dim()
        d_dim = self.doc_encoder.get_output_dim()
        # set up batchnorm
        self.query_batch_norm = nn.BatchNorm1d(q_dim)
        self.document_batch_norm = nn.BatchNorm1d(d_dim)

        input_dim = (q_dim + d_dim)
        if use_encoded: input_dim *= 2
        # set up the scorer
        if scorer is None:
            scorer = FeedForward(
                        input_dim=input_dim, num_layers=1,
                        hidden_dims=1, activations=Activation.by_name('linear')(), dropout=0.)
        self.scorer = scorer
        # assertions to ensure our shapes match our assumptions
        assert q_dim == d_dim
        assert self.scorer.get_output_dim() == 1
        assert self.scorer.get_input_dim() == input_dim

    def forward(self,
                query: torch.Tensor,
                document: torch.Tensor,
                query_mask: torch.Tensor = None,
                document_mask: torch.Tensor = None) -> torch.Tensor:

        q = self.query_encoder(query, query_mask)
        q = self.query_batch_norm(q)

        if self.attention is not None:
            attn = self.attention(q, document, document_mask).unsqueeze(2)
            document = document * attn

        d = self.doc_encoder(document, document_mask)
        d = self.document_batch_norm(d)

        reprs = [q - d, q * d]
        if self.use_encoded:
            reprs += [q, d]

        encoded = torch.cat(reprs, dim=1)

        return self.scorer(encoded)
