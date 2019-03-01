import torch

from .scorer import Scorer
from typing import Optional
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.modules.attention import Attention

class DecomposableAttention(Attention):
    def __init__(self) -> None: pass

    def forward(self, Q: torch.Tensor, V: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        pass

@Scorer.register("decomposable_scorer")
class DecomposableScorer(Scorer):
    """
    The ``DecomposableScorer'' is a ``Scorer'' which uses decomposable attention
    (https://arxiv.org/abs/1606.01933) to compute a relevance score for the
    document/query pair.

    Parameters
    ----------

    """

    def __init__(self) -> None:
        pass

    def forward(self,
                query: torch.Tensor,
                document: torch.Tensor,
                query_mask: torch.Tensor = None,
                document_mask: torch.Tensor = None) -> torch.Tensor:
        return torch.Tensor([1.])
