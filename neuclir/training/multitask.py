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
from allennlp.common.util import (dump_metrics, gpu_memory_mb, peak_memory_mb,
                                  get_frozen_and_tunable_parameter_names, lazy_groups_of)
from allennlp.common.checks import ConfigurationError

from typing import List, Dict, Iterable, Any, Set, Optional, NamedTuple

import itertools
import logging
import torch
import tqdm
import os

logger = logging.getLogger(__name__)
#
# def flatten_datasets(datasets: Dict[str, Iterable[Instance]],
#                      dataset_name_field: Optional[str] = 'dataset') -> Iterable[Instance]:
#     for name, iterator in datasets.items():
#         for instance in iterator:
#             instance.fields[dataset_name_field] = MetadataField(name)
#             yield instance

def bad_chain(a: Iterable[Any], b: Iterable[Any]) -> Iterable[Any]:
    a_prime = itertools.tee(a)
    b_prime = itertools.tee(b)
    for i in a_prime:
        yield i
    for i in b_prime:
        yield i

def load_datasets(data_paths: Dict[str, str], readers: Dict[str, DatasetReader]) -> Iterable[Instance]:
    data: Iterable[Instance] = []

    for name, data_path in data_paths.items():
        logger.info("Reading training data from %s", data_path)
        if name not in readers:
            raise ConfigurationError(f"Dataset reader {name} not found in readers")
        data.extend(readers[name].read(data_path))

    return data


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

    datasets: Dict[str, Iterable[Instance]] = {
        "train": load_datasets(train_data_paths, readers)
    }

    if validation_data_paths is not None:
        datasets["validation"] = load_datasets(validation_data_paths, readers)

    if test_data_paths is not None:
        datasets["test"] = load_datasets(test_data_paths, readers)

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

        train_data = all_datasets['train']
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

        base = Trainer.from_params(model=pieces.model,
                                   serialization_dir=serialization_dir,
                                   iterator=pieces.iterator,
                                   train_data=pieces.train_dataset,
                                   validation_data=pieces.validation_dataset,
                                   params=pieces.params,
                                   validation_iterator=pieces.validation_iterator)
        base.__class__ = MultiTaskTrainer

        return base
