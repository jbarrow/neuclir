import torch

from allennlp.common import Registrable

class Scorer(torch.nn.Module, Registrable):
    """
    A ``Scorer'' is a ``Module'' that takes a (query, document) pair
    as input and assigns a score.
    """
    def get_output_dim(self) -> int:
        """
        Returns the dimension of the score.
        """
        return 1
