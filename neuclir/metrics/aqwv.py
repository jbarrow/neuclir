import json
import torch
import numpy as np

from typing import Optional
from overrides import overrides
from sklearn.metrics import confusion_matrix
from allennlp.training.metrics.metric import Metric
from .utils import paired_sort

@Metric.register('aqwv')
class AQWV(Metric):
    def __init__(self, corrections_file: Optional[str] = None,
                 beta: float = 40., cutoff: int = 40, version: str = 'tuning') -> None:
        super(AQWV, self).__init__()

        self.beta = beta
        self.cutoff = cutoff
        self.version = version
        self.q_ignored_relevant = 0
        self.q_no_scores = 0

        if corrections_file is not None:
            with open(corrections_file) as fp:
                for line in fp:
                    q = json.loads(line)
                    self.q_no_scores += 1
                    if q['ignored_relevant'] > 0:
                        self.q_ignored_relevant += 1

        self.miss = [1.,]*self.q_ignored_relevant
        self.false_alarm = [0.,]*(self.q_no_scores)


    def __call__(self, outputs: torch.Tensor,
                 targets: torch.LongTensor,
                 masks: torch.LongTensor = None,
                 relevant_ignored: torch.LongTensor = None,
                 irrelevant_ignored: torch.LongTensor = None) -> None:

        if not len(outputs.shape) == len(targets.shape):
            targets = torch.unsqueeze(targets, 1)
            targets = torch.zeros_like(outputs).scatter_(1, targets, 1)

        if relevant_ignored is None:
            relevant_ignored = np.zeros(outputs.shape[0])
        else:
            relevant_ignored = relevant_ignored.cpu().numpy()

        if irrelevant_ignored is None:
            irrelevant_ignored = np.zeros(outputs.shape[0])
        else:
            irrelevant_ignored = irrelevant_ignored.cpu().numpy()

        if masks is None:
            masks = torch.ones_like(targets).int()
        else:
            # try converting to int
            masks = masks.int()

        for output, target, mask, rel_ignored, irrel_ignored in zip(outputs,
                                                                    targets,
                                                                    masks,
                                                                    relevant_ignored,
                                                                    irrelevant_ignored):
            valid_docs = torch.sum(mask)
            output, target = output[:valid_docs], target[:valid_docs]

            output, target = paired_sort(output, target)

            output = self._cutoff(output)

            confusion = self.confusion_matrix(output, target)

            if torch.sum(target).cpu().numpy() + rel_ignored == 0:
                if self.version == 'tuning':
                    # ignore both miss and false alarm when tuning
                    pass
                elif self.version == 'program':
                    # ignore miss when calculating program target
                    false_alarm = confusion[0, 1] / float(sum(confusion[0, :]) + irrel_ignored)
                    self.false_alarm.append(false_alarm)
            else:
                miss = (confusion[1, 0] + rel_ignored) / float(sum(confusion[1, :]) + rel_ignored)
                false_alarm = confusion[0, 1] / float(sum(confusion[0, :]) + irrel_ignored)
                self.false_alarm.append(false_alarm)
                self.miss.append(miss)

    def confusion_matrix(self, prediction, target):
        prediction = prediction.cpu().numpy()
        target = target.cpu().numpy()
        return confusion_matrix(target, prediction, labels=[0, 1])

    def _cutoff(self, output):
        doc_lens = output.shape[0]
        return torch.arange(doc_lens).lt(self.cutoff).type_as(output)

    def get_metric(self, reset: bool = False) -> float:
        if len(self.false_alarm) == 0:
            aqwv = 0.0
        else:
            #print('misses:', self.miss)
            #print('false_alarms:', self.false_alarm)
            aqwv = 1 - np.mean(self.miss) - self.beta * np.mean(self.false_alarm)
        if reset:
            self.reset()
        return aqwv

    @overrides
    def reset(self) -> None:
        self.miss = [1.,]*self.q_ignored_relevant
        self.false_alarm = [0.,]*(self.q_no_scores)
