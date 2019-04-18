from allennlp.training.optimizers import Optimizer
from allennlp.training.trainer_base import TrainerBase
from allennlp.data.iterators import DataIterator
from allennlp.training.checkpointer import Checkpointer
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.params import Params
from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from ..data.minglers import DatasetMingler

from typing import List, Dict, Iterable, Any, Set

import torch
import tqdm


@TrainerBase.register("multi-task-trainer")
class MultiTaskTrainer(TrainerBase):
    """
    A simple trainer that works in our multi-task setup.
    Really the main thing that makes this task not fit into our
    existing trainer is the multiple datasets.
    """
    def __init__(self,
                 model: Model,
                 serialization_dir: str,
                 iterator: DataIterator,
                 mingler: DatasetMingler,
                 optimizer: torch.optim.Optimizer,
                 datasets: Dict[str, Iterable[Instance]],
                 num_epochs: int = 10,
                 num_serialized_models_to_keep: int = 10) -> None:
        super().__init__(serialization_dir)
        self.model = model
        self.iterator = iterator
        self.mingler = mingler
        self.optimizer = optimizer
        self.datasets = datasets
        self.num_epochs = num_epochs
        self.checkpointer = Checkpointer(serialization_dir,
                                         num_serialized_models_to_keep=num_serialized_models_to_keep)

    def save_checkpoint(self, epoch: int) -> None:
        training_state = {"epoch": epoch, "optimizer": self.optimizer.state_dict()}
        self.checkpointer.save_checkpoint(epoch, self.model.state_dict(), training_state, True)

    def restore_checkpoint(self) -> int:
        model_state, trainer_state = self.checkpointer.restore_checkpoint()
        if not model_state and not trainer_state:
            return 0
        else:
            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(trainer_state["optimizer"])
            return trainer_state["epoch"] + 1


    def train(self) -> Dict:
        start_epoch = self.restore_checkpoint()

        self.model.train()
        for epoch in range(start_epoch, self.num_epochs):
            total_loss = 0.0
            batches = tqdm.tqdm(self.iterator(self.mingler.mingle(self.datasets), num_epochs=1))
            for i, batch in enumerate(batches):
                self.optimizer.zero_grad()
                loss = self.model.forward(**batch)['loss']
                loss.backward()
                total_loss += loss.item()
                self.optimizer.step()
                batches.set_description(f"epoch: {epoch} loss: {total_loss / (i + 1)}")

            # Save checkpoint
            self.save_checkpoint(epoch)

        return {}

    @classmethod
    def from_params(cls,   # type: ignore
                    params: Params,
                    serialization_dir: str,
                    recover: bool = False) -> 'MultiTaskTrainer':
        readers = {name: DatasetReader.from_params(reader_params)
                   for name, reader_params in params.pop("train_dataset_readers").items()}
        train_file_paths = params.pop("train_file_paths").as_dict()

        datasets = {name: reader.read(train_file_paths[name])
                    for name, reader in readers.items()}

        instances = (instance for dataset in datasets.values() for instance in dataset)
        vocab = Vocabulary.from_params(Params({}), instances)
        model = Model.from_params(params.pop('model'), vocab=vocab)
        iterator = DataIterator.from_params(params.pop('iterator'))
        iterator.index_with(vocab)
        mingler = DatasetMingler.from_params(params.pop('mingler'))

        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(parameters, params.pop('optimizer'))

        num_epochs = params.pop_int("num_epochs", 10)

        _ = params.pop("trainer", Params({}))

        params.assert_empty(__name__)

        return MultiTaskTrainer(model, serialization_dir, iterator, mingler, optimizer, datasets, num_epochs)
