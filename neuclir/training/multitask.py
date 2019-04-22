from allennlp.training.optimizers import Optimizer
from allennlp.training.trainer import Trainer
from allennlp.training.trainer_base import TrainerBase
from allennlp.data.iterators import DataIterator
from allennlp.training.checkpointer import Checkpointer
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.params import Params
from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.fields import MetadataField
from allennlp.common.util import (dump_metrics, gpu_memory_mb, peak_memory_mb,
                                  get_frozen_and_tunable_parameter_names, lazy_groups_of)

from typing import List, Dict, Iterable, Any, Set, Optional, NamedTuple

import logging
import torch
import tqdm
import os

logger = logging.getLogger(__name__)

def flatten_datasets(datasets: Dict[str, Iterable[Instance]],
                     dataset_name_field: Optional[str] = 'dataset') -> Iterable[Instance]:
    for name, iterator in datasets.items():
        for instance in iterator:
            instance.fields[dataset_name_field] = MetadataField(name)
            yield instance

def multitask_datasets_from_params(params: Params) -> Dict[str, Iterable[Instance]]:
    """
    Load all the datasets specified by the config.
    """
    # In the multitask setting, the dataset types are indexed by the names. As
    # such, you can use a disjoint set of readers for the train, test, and
    # validation sets (just give them different names)
    readers = {name: DatasetReader.from_params(reader_params)
               for name, reader_params in params.pop("dataset_readers").items()}

    train_data_paths = params.pop('train_data_paths')
    validation_data_paths = params.pop('validation_data_paths', None)
    test_data_paths = params.pop("test_data_paths", None)

    train_data: Dict[str, Iterable[Instance]] = {}
    for name, train_data_path in train_data_paths.items():
        logger.info("Reading training data from %s", train_data_path)
        if name not in readers:
            raise ConfigurationError(f"Dataset reader {name} not found in readers")
        train_data[name] = readers[name].read(train_data_path)
    datasets: Dict[str, Iterable[Instance]] = {
        "train": list(flatten_datasets(train_data))
    }

    if validation_data_paths is not None:
        validation_data: Dict[str, Iterable[Instance]] = {}
        for name, validation_data_path in validation_data_paths.items():
            logger.info("Reading validation data from %s", validation_data_path)
            if name not in readers:
                raise ConfigurationError(f"Dataset reader {name} not found in readers")
            validation_data[name] = readers[name].read(validation_data_path)
        datasets["validation"] = flatten_datasets(validation_data)

    if test_data_paths is not None:
        test_data: Dict[str, Iterable[Instance]] = {}
        for name, test_data_path in test_data_paths.items():
            logger.info("Reading test data from %s", test_data_path)
            if name not in readers:
                raise ConfigurationError(f"Dataset reader {name} not found in readers")
            test_data[name] = readers[name].read(test_data_path)
        datasets["test"] = flatten_datasets(test_data)

    return datasets


class MultiTaskTrainerPieces(NamedTuple):
    """
    We would like to avoid having complex instantiation logic taking place
    in `Trainer.from_params`. This helper class has a `from_params` that
    instantiates a model, loads train (and possibly validation and test) datasets,
    constructs a Vocabulary, creates data iterators, and handles a little bit
    of bookkeeping. If you're creating your own alternative training regime
    you might be able to use this.
    """
    model: Model
    iterator: DataIterator
    train_dataset: Iterable[Instance]
    validation_dataset: Iterable[Instance]
    test_dataset: Iterable[Instance]
    validation_iterator: DataIterator
    params: Params

    @staticmethod
    def from_params(params: Params, serialization_dir: str, recover: bool = False) -> 'TrainerPieces':
        all_datasets = multitask_datasets_from_params(params)
        datasets_for_vocab_creation = set(params.pop("datasets_for_vocab_creation", all_datasets))

        for dataset in datasets_for_vocab_creation:
            if dataset not in all_datasets:
                raise ConfigurationError(f"invalid 'dataset_for_vocab_creation' {dataset}")

        logger.info("From dataset instances, %s will be considered for vocabulary creation.",
                    ", ".join(datasets_for_vocab_creation))

        if recover and os.path.exists(os.path.join(serialization_dir, "vocabulary")):
            vocab = Vocabulary.from_files(os.path.join(serialization_dir, "vocabulary"))
            params.pop("vocabulary", {})
        else:
            vocab = Vocabulary.from_params(
                    params.pop("vocabulary", {}),
                    (instance for key, dataset in all_datasets.items()
                     for instance in dataset
                     if key in datasets_for_vocab_creation)
            )

        model = Model.from_params(vocab=vocab, params=params.pop('model'))

        # If vocab extension is ON for training, embedding extension should also be
        # done. If vocab and embeddings are already in sync, it would be a no-op.
        model.extend_embedder_vocab()

        # Initializing the model can have side effect of expanding the vocabulary
        vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))

        iterator = DataIterator.from_params(params.pop("iterator"))
        iterator.index_with(model.vocab)
        validation_iterator_params = params.pop("validation_iterator", None)
        if validation_iterator_params:
            validation_iterator = DataIterator.from_params(validation_iterator_params)
            validation_iterator.index_with(model.vocab)
        else:
            validation_iterator = None

        train_data = list(all_datasets['train'])
        validation_data = all_datasets.get('validation')
        test_data = all_datasets.get('test')

        trainer_params = params.pop("trainer")
        no_grad_regexes = trainer_params.pop("no_grad", ())
        for name, parameter in model.named_parameters():
            if any(re.search(regex, name) for regex in no_grad_regexes):
                parameter.requires_grad_(False)

        frozen_parameter_names, tunable_parameter_names = \
                    get_frozen_and_tunable_parameter_names(model)
        logger.info("Following parameters are Frozen  (without gradient):")
        for name in frozen_parameter_names:
            logger.info(name)
        logger.info("Following parameters are Tunable (with gradient):")
        for name in tunable_parameter_names:
            logger.info(name)

        return MultiTaskTrainerPieces(model, iterator,
                             train_data, validation_data, test_data,
                             validation_iterator, trainer_params)

@TrainerBase.register("multi-task-trainer")
class MultiTaskTrainer(Trainer):
    @classmethod
    def from_params(cls, # type: ignore
                    params: Params,
                    serialization_dir: str,
                    recover: bool = False) -> 'MultiTaskTrainer':
        pieces = MultiTaskTrainerPieces.from_params(params, serialization_dir, recover)  # pylint: disable=no-member
        print(len(list(pieces.train_dataset)))
        base = Trainer.from_params(model=pieces.model,
                                   serialization_dir=serialization_dir,
                                   iterator=pieces.iterator,
                                   train_data=pieces.train_dataset,
                                   validation_data=pieces.validation_dataset,
                                   params=pieces.params,
                                   validation_iterator=pieces.validation_iterator)
        base.__class__ = MultiTaskTrainer
        return base

    """
    A simple trainer that works in our multi-task setup.
    Really the main thing that makes this task not fit into our
    existing trainer is the multiple datasets.
    """
    # def __init__(self,
    #              model: Model,
    #              serialization_dir: str,
    #              iterator: DataIterator,
    #              mingler: DatasetMingler,
    #              optimizer: torch.optim.Optimizer,
    #              datasets: Dict[str, Iterable[Instance]],
    #              num_epochs: int = 10,
    #              num_serialized_models_to_keep: int = 10) -> None:
    #     super().__init__(serialization_dir)
    #     self.model = model
    #     self.iterator = iterator
    #     self.mingler = mingler
    #     self.optimizer = optimizer
    #     self.datasets = datasets
    #     self.num_epochs = num_epochs
    #     self.checkpointer = Checkpointer(serialization_dir,
    #                                      num_serialized_models_to_keep=num_serialized_models_to_keep)
    #
    # def save_checkpoint(self, epoch: int) -> None:
    #     training_state = {"epoch": epoch, "optimizer": self.optimizer.state_dict()}
    #     self.checkpointer.save_checkpoint(epoch, self.model.state_dict(), training_state, True)
    #
    # def restore_checkpoint(self) -> int:
    #     model_state, trainer_state = self.checkpointer.restore_checkpoint()
    #     if not model_state and not trainer_state:
    #         return 0
    #     else:
    #         self.model.load_state_dict(model_state)
    #         self.optimizer.load_state_dict(trainer_state["optimizer"])
    #         return trainer_state["epoch"] + 1
    #
    #
    # def train(self) -> Dict:
    #     start_epoch = self.restore_checkpoint()
    #
    #     self.model.train()
    #     for epoch in range(start_epoch, self.num_epochs):
    #         total_loss = 0.0
    #         batches = tqdm.tqdm(self.iterator(self.mingler.mingle(self.datasets), num_epochs=1))
    #         for i, batch in enumerate(batches):
    #             self.optimizer.zero_grad()
    #             loss = self.model.forward(**batch)['loss']
    #             loss.backward()
    #             total_loss += loss.item()
    #             self.optimizer.step()
    #             batches.set_description(f"epoch: {epoch} loss: {total_loss / (i + 1)}")
    #
    #         # Save checkpoint
    #         self.save_checkpoint(epoch)
    #
    #     return {}

    # @classmethod
    # def from_params(cls,   # type: ignore
    #                 params: Params,
    #                 serialization_dir: str,
    #                 recover: bool = False) -> 'MultiTaskTrainer':
    #    readers = {name: DatasetReader.from_params(reader_params)
    #               for name, reader_params in params.pop("train_dataset_readers").items()}
    #     train_file_paths = params.pop("train_file_paths").as_dict()
    #
    #     datasets = {name: reader.read(train_file_paths[name])
    #                 for name, reader in readers.items()}
    #
    #     instances = (instance for dataset in datasets.values() for instance in dataset)
    #     vocab = Vocabulary.from_params(Params({}), instances)
    #     model = Model.from_params(params.pop('model'), vocab=vocab)
    #     iterator = DataIterator.from_params(params.pop('iterator'))
    #     iterator.index_with(vocab)
    #     mingler = DatasetMingler.from_params(params.pop('mingler'))
    #
    #     parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
    #     optimizer = Optimizer.from_params(parameters, params.pop('optimizer'))
    #
    #     num_epochs = params.pop_int("num_epochs", 10)
    #
    #     _ = params.pop("trainer", Params({}))
    #
    #     params.assert_empty(__name__)
    #
    #     return MultiTaskTrainer(model, serialization_dir, iterator, mingler, optimizer, datasets, num_epochs)
