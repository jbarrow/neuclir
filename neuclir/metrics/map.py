import copy
import json
import torch
import numpy as np

from typing import Optional
from overrides import overrides
from allennlp.training.metrics.metric import Metric
from .utils import paired_sort

@Metric.register('map')
class MeanAveragePrecision(Metric):
    def __init__(self, k: Optional[int] = 1000, corrections_file: Optional[str] = None):
        super(MeanAveragePrecision, self).__init__()

        self.k = k

        if corrections_file is not None:
            with open(corrections_file) as fp:
                self.missed = len([l for l in fp])
                self.base_aps = [0.,] * self.missed
        else:
            self.base_aps = []

        self.aps = copy.deepcopy(self.base_aps)

    def __call__(self, outputs: torch.Tensor,
                 targets: torch.LongTensor,
                 masks: Optional[torch.LongTensor] = None,
                 relevant_ignored: Optional[torch.LongTensor] = None,
                 irrelevant_ignored: Optional[torch.LongTensor] = None):

        if not len(outputs.shape) == len(targets.shape):
            targets = torch.unsqueeze(targets, 1)
            targets = torch.zeros_like(outputs).scatter_(1, targets, 1)

        if relevant_ignored is None:
            relevant_ignored = np.zeros(outputs.shape[0])
        else:
            relevant_ignored = relevant_ignored.cpu().numpy()

        if masks is None:
            masks = torch.ones_like(targets).int()
        else:
            # try converting to int
            masks = masks.int()

        for output, target, mask, rel in zip(outputs, targets, masks, relevant_ignored):
            valid_docs = torch.sum(mask)
            output, target = output[:valid_docs], target[:valid_docs]
            output, target = paired_sort(output, target)
            predicted = np.arange(output.shape[0])
            actual = set(torch.nonzero(target).view(-1).cpu().numpy().tolist())
            self.aps.append(self.apk(actual, predicted, rel))

    def apk(self, actual, predicted, rel):
        if len(predicted) > self.k:
            predicted = predicted[:self.k]

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p in actual:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        if not actual:
            return 0.0

        return score / (min(len(actual), self.k) + rel)

    def get_metric(self, reset: bool = False):
        if len(self.aps) == 0:
            map = 0.0
        else:
            map = np.mean(self.aps)
        if reset:
            self.reset()
        return map

    @overrides
    def reset(self):
        self.aps = copy.deepcopy(self.base_aps)
