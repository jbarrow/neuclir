import torch

from typing import Tuple

def paired_sort(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    sorted_a, indices = torch.sort(a, descending=True)
    sorted_b = b[indices]
    return sorted_a, sorted_b
