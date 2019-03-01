import torch

from allennlp.common import Registrable

class Loss(torch.nn.Module, Registrable):
    """
    A ``Loss'' is both ``Registrable'' and a ``Module'' that wraps a PyTorch
    loss function -- i.e. it returns a scalar.
    """
    def get_output_dim(self) -> int:
        """
        Returns the dimension of the score.
        """
        return 1
