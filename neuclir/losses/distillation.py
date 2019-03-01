from .loss import Loss


class DistillationLoss(Loss):
    """
    ``DistillationLoss'' takes a temperature
    """

    def __init__(self, T : float = 1.0):
        super(DistillationLoss, self ).__init__()
