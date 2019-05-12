import torch
import torch.nn as nn

from .scorer import Scorer
from typing import Optional
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.nn import Activation

@Scorer.register("dssm_scorer")
class DSSMScorer(Scorer):
    """
    The ``DSSMScorer'' is a ``Scorer'' which computes a single vector
    representation for each query and document, and then computes the

    Parameters
    ----------
    encoder: ``Seq2VecEncoder''
        This is the encoder for the document. Common options include the
        ``BagOfEmbeddingsEncoder'' or the ``CNNEncoder''.
    transformer: ``FeedForward''
        If no feedforward scorer is provided, we use a single-layer feedforward
        network that uses.
    """

    def __init__(self,
                 encoder: Seq2VecEncoder,
                 transformer: FeedForward) -> None:

        super(DSSMScorer, self).__init__()

        self._encoder = encoder
        self._transformer = transformer
        self._similarity = nn.CosineSimilarity()

    def forward(self,
                query: torch.Tensor,
                document: torch.Tensor,
                query_mask: torch.Tensor = None,
                document_mask: torch.Tensor = None) -> torch.Tensor:

        q = self._encoder(query, query_mask)
        q = self._transformer(q)

        d = self._encoder(document, document_mask)
        d = self._transformer(d)

        return self._similarity(q, d)
