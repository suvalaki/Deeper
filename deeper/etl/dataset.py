from __future__ import annotations
from abc import (
    ABC,
    abstractmethod,
    abstractproperty,
    abstractclassmethod,
)
from typing import NamedTuple, Tuple, Generic, Sequence, TypeVar, Callable
from dataclasses import dataclass, asdict
import numpy as np
import dill as pickle
import pandas as pd
import numpy as np
from sqlalchemy.sql.sqltypes import Boolean
import tensorflow as tf
import json

from brownlow.etl.cache.getter_base import OutputData
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from math import floor


def divisibale_by(n, d):
    if n % d == 0:
        return n
    else:
        return divisibale_by(n - 1, d)


def isinstance_namedtuple(obj) -> bool:
    return isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields")


@dataclass
class TrainValidTestData:
    ds_train_noshuffle: tf.data.Dataset
    ds_train_valid: tf.data.Dataset
    ds_train: tf.data.Dataset
    ds_valid: tf.data.Dataset
    ds_test: tf.data.Dataset
    train_valid_data: list
    train_data: list
    valid_data: list
    test_data: list
    training_valid_idx: Tuple[int, ...]
    training_idx: Tuple[int, ...]
    valid_idx: Tuple[int, ...]
    testset_idx: Tuple[int, ...]

    training_valid_gen: Callable
    training_gen: Callable
    valid_gen: Callable
    test_gen: Callable


@dataclass
class KFoldTrainValidTestData:
    folds: int
    ds_train_noshuffle: tf.data.Dataset
    ds_train_valid: tf.data.Dataset
    ds_train: Tuple[tf.data.Dataset, ...]
    ds_valid: Tuple[tf.data.Dataset, ...]
    ds_test: tf.data.Dataset
    train_valid_data: list
    train_data: Tuple[list, ...]
    valid_data: Tuple[list, ...]
    test_data: list
    training_valid_idx: Tuple[int, ...]
    training_idx: Tuple[Tuple[int, ...]]
    valid_idx: Tuple[Tuple[int, ...]]
    testset_idx: Tuple[int, ...]

    training_valid_gen: Callable
    training_gen: Tuple[Callable, ...]
    valid_gen: Tuple[Callable, ...]
    test_gen: Callable


class TensorSpecTuple:

    # The expectation is that all the named tuples are to implement the
    # tensor_spec method defined bellow here. The are to implement:
    #
    #   @property #Boilerplate
    #   def tensor_spec(self):
    #       return TensorSpecTuple.tensor_spec(self)
    #
    # TODO: Validate whether we could just be using dataclasses with
    # inheritance instead.

    @staticmethod
    def option(x):

        if x is None:
            return tf.TensorSpec(())
        if isinstance(x, (int, float, str)):
            return tf.TensorSpec((1,))
        if isinstance(x, (pd.Series, pd.DataFrame, np.ndarray)):
            return tf.TensorSpec(x.shape)
        if isinstance_namedtuple(x):
            return x.tensor_spec
        if isinstance(x, (list, tuple)):
            # print(type(x))
            return tuple([z.tensor_spec for z in x])

    @staticmethod
    def printable_option(x):

        if x is None:
            return "NONE"
        if isinstance(x, (int, float, str)):
            return tf.TensorSpec((1,))
        if isinstance(x, (pd.Series, pd.DataFrame, np.ndarray)):
            return tf.TensorSpec(x.shape)
        if isinstance_namedtuple(x):
            return x.pretty_tensor_spec
        if isinstance(x, (list, tuple)):
            print(type(x))
            return (
                "[\n  "
                + "\n  ".join(
                    [
                        f"{i}: " + str(z.pretty_tensor_spec).replace("\n", "\n  ") + ","
                        for i, z in enumerate(x)
                    ]
                )
                + "\n]"
            )

    @classmethod
    def tensor_spec(cls, tpe):
        return type(tpe)(*[cls.option(x) for x in tpe])

    @classmethod
    def printable_tensor_spec(cls, tpe):
        return (
            str(tpe.__class__.__name__)
            + "(\n  "
            + ",\n  ".join(
                [
                    k + ": " + str(cls.printable_option(x)).replace("\n", "\n  ")
                    for k, x in tpe._asdict().items()
                ]
            )
            + "\n)"
        )


@dataclass
class PipelineData(ABC):
    input_cache: Sequence[NamedTuple] = None
    transform_cache: Sequence[NamedTuple] = None


@dataclass
class PipelineConfig(ABC):
    @abstractproperty
    def save_signature(self) -> str:
        pass

    def to_json(self, filename: str):
        with open(filename, "w") as outfile:
            json.dump(asdict(self), outfile)

    @classmethod
    def read_json(cls, filename: str):
        return cls(**json.load(filename))


class ModelInputData(NamedTuple):
    @abstractclassmethod
    def from_implicit_getter():
        ...


DatasetConfig = TypeVar("DatasetConfig", bound=PipelineConfig)
ModelInputDataType = TypeVar("ModelInputDataType", bound=ModelInputData)
ModelInputDataTypeX = TypeVar("ModelInputDataTypeX", bound=ModelInputData)
ModelInputDataTypeY = TypeVar("ModelInputDataTypeY", bound=ModelInputData)


class Dataset(
    ABC,
    Generic[
        DatasetConfig,
        OutputData,
        ModelInputDataType,
        ModelInputDataTypeX,
        ModelInputDataTypeY,
    ],
    tf.keras.utils.Sequence,
):

    # To be replaced by the target types in child definitions
    input_type: TypeVar = OutputData
    output_type: TypeVar = ModelInputDataType
    output_type_x: TypeVar = ModelInputDataTypeX
    output_type_y: TypeVar = ModelInputDataTypeY
    _data: PipelineData = PipelineData()
    _config: PipelineConfig = None

    def __init__(self):
        super().__init__()

    @abstractmethod
    def _output_type_from_implicit_getter(self, x):
        pass

    @abstractmethod
    def run(self):
        pass

    @property
    def config(self) -> DatasetConfig:
        return self._config

    @property
    def save_signature(self) -> str:
        return self._config.save_signature

    def save(self, fp: str = None):
        pickle.dump(
            (self._data, self._config),
            open(fp or self._config.save_signature, "wb"),
        )

    def load(self, fp: str = None) -> None:
        self._data, self._config = pickle.load(open(fp or self._config.save_signature, "rb"))

    @abstractmethod
    def _cache_data(self):
        """Extract using the data getter the input data into this transformer"""
        pass

    @abstractclassmethod
    def _transform_pipeline(self, x: OutputData) -> OutputData:
        """Transform a single datum of the extracted data"""
        ...

    # @abstractmethod
    def to_numpy(self) -> Tuple[np.array, np.array]:
        pass

    def _cache_transformed_data(self):
        self._data.transform_cache = [
            self._output_type_from_implicit_getter(self._transform_pipeline(x))
            for x in self._data.input_cache
        ]

    def _get_first_cached_transformed(self):
        try:
            return self._data.transform_cache[0]
        except:
            return self._output_type_from_implicit_getter(self._data.input_cache[0])

    @property
    def input_signature(self) -> ModelInputDataTypeX:
        return self._get_first_cached_transformed().X.tensor_spec

    @property
    def output_signature(self) -> ModelInputDataTypeY:
        return self._get_first_cached_transformed().Y.tensor_spec

    @property
    def eval_signature(self) -> ModelInputDataType:
        return self._get_first_cached_transformed().tensor_spec

    @property
    def pretty_eval_signature(self) -> ModelInputDataType:
        return self._get_first_cached_transformed().pretty_tensor_spec

    def __iter__(self):
        self._iter_counter = 0
        return self

    def __next__(self) -> ModelInputDataType:

        if self._iter_counter < len(self._data.input_cache):
            if self._data.transform_cache:
                val = self._data.transform_cache[self._iter_counter]
            else:
                val = self._output_type_from_implicit_getter(
                    self._transform_pipeline(self._data.input_cache[self._iter_counter]),
                )
            self._iter_counter += 1
            return val
        self._iter_counter = 0
        raise StopIteration()

    def __call__(self):
        return self.__iter__()

    def __len__(self):
        return len(self._data.transform_cache)

    def __getitem__(self, index):
        try:
            return self._data.transform_cache[index]
        except:
            return self._output_type_from_implicit_getter(
                self._transform_pipeline(self._data.input_cache[index]),
            )

    def _stratification_lambda(self, x: ModelInputDataType) -> bool:
        return True

    def _testset_stratification_lambda(self, x: ModelInputDataType) -> bool:
        return False

    def create_dataset(
        self,
        batch_size=1,
    ):

        data = [d for d in self]
        n = divisibale_by(len(data), batch_size)
        # data = data[:n]

        def data_gen():
            for xy in data:
                yield xy

        ds = (
            tf.data.Dataset.from_generator(data_gen, output_signature=self.eval_signature)
            .take(n)
            .batch(batch_size)
            .cache()
        )
        return ds

    def _assign_train_test_idx(self, train_split, batch_size, seed):
        idx = list(range(len(self)))

        # pull out testset
        testset_bool = [self._testset_stratification_lambda(x) for x in self]
        testset_idx = [i for i, t in enumerate(testset_bool) if t]
        test_data = [x for t, x in zip(testset_bool, self) if t]

        train_valid_idx = [i for i in range(len(self)) if i not in testset_idx]
        train_valid_data = [x for i, x in enumerate(self) if i not in testset_idx]
        train_valid_strat = [self._stratification_lambda(x) for x in train_valid_data]

        nData = len(train_valid_idx)
        n_train = divisibale_by(floor(train_split * nData), batch_size)
        training_idx, valid_idx, train_data, valid_data = train_test_split(
            train_valid_idx,
            train_valid_data,
            train_size=n_train,
            stratify=train_valid_strat,
            random_state=seed,
        )

        def training_data_gen():
            for xy in train_data:
                yield xy

        def validation_data_gen():
            for xy in valid_data:
                yield xy

        def testing_data_gen():
            for xy in test_data:
                yield xy

        return (
            train_valid_data,
            train_data,
            valid_data,
            test_data,
            train_valid_idx,
            training_idx,
            valid_idx,
            testset_idx,
        )

    def create_train_test_dataset(
        self,
        train_split,
        batch_size=1,
        test_batch_size=1,
        shuffle_training=True,
        seed=1234,
    ):

        data = [d for d in self]
        all_data_cat = [self._stratification_lambda(x) for x in data]
        nData = len(data)
        n_train = divisibale_by(floor(train_split * nData), batch_size)
        n_test = divisibale_by(nData - n_train, test_batch_size)

        training_data, testing_data = train_test_split(
            data, train_size=n_train, stratify=all_data_cat, random_state=seed
        )
        testing_data = testing_data[:n_test]

        def training_data_gen():
            for xy in training_data:
                yield xy

        def testing_data_gen():
            for xy in testing_data:
                yield xy

        ds_train = (
            tf.data.Dataset.from_generator(training_data_gen, output_signature=self.eval_signature)
            # .take(n_train)
            .shuffle(n_train, reshuffle_each_iteration=shuffle_training)
            .batch(batch_size)
            .cache()
        )
        ds_test = (
            tf.data.Dataset.from_generator(testing_data_gen, output_signature=self.eval_signature)
            # .take(n_test)
            .batch(test_batch_size).cache()
        )

        return ds_train, ds_test

    def create_train_valid_test_dataset(
        self,
        train_split,
        batch_size=1,
        test_batch_size=1,
        shuffle_training=True,
        seed=1234,
        oversample: bool = True,
    ) -> TrainValidTestData:

        (
            train_valid_data,
            train_data,
            valid_data,
            test_data,
            train_valid_idx,
            training_idx,
            valid_idx,
            testset_idx,
        ) = self._assign_train_test_idx(train_split, batch_size, seed)

        def training_valid_data_gen():
            for xy in train_valid_data:
                yield xy

        def training_data_gen():
            for xy in train_data:
                yield xy

        def validation_data_gen():
            for xy in valid_data:
                yield xy

        def testing_data_gen():
            for xy in test_data:
                yield xy

        ds_train_valid = (
            tf.data.Dataset.from_generator(
                training_valid_data_gen, output_signature=self.eval_signature
            )
            # .take(n_train)
            .batch(batch_size)
        )
        ds_train_noshuffle = (
            tf.data.Dataset.from_generator(training_data_gen, output_signature=self.eval_signature)
            # .take(n_train)
            .batch(batch_size)
        )

        if oversample:
            train_data_pos = [xy for xy in train_data if self._stratification_lambda(xy)]
            train_data_neg = [xy for xy in train_data if not self._stratification_lambda(xy)]

            def training_data_gen_pos():
                for xy in train_data_pos:
                    yield xy

            def training_data_gen_neg():
                for xy in train_data_neg:
                    yield xy

            ds_train_pos = (
                tf.data.Dataset.from_generator(
                    training_data_gen_pos, output_signature=self.eval_signature
                )
                # .take(n_train)
                .shuffle(len(train_data), reshuffle_each_iteration=shuffle_training)
            )
            ds_train_neg = (
                tf.data.Dataset.from_generator(
                    training_data_gen_neg, output_signature=self.eval_signature
                )
                # .take(n_train)
                .shuffle(len(train_data), reshuffle_each_iteration=shuffle_training)
            )
            ds_train = tf.data.Dataset.sample_from_datasets(
                [ds_train_pos, ds_train_neg], weights=[0.5, 0.5]
            ).batch(batch_size)

        else:
            ds_train = (
                tf.data.Dataset.from_generator(
                    training_data_gen, output_signature=self.eval_signature
                )
                # .take(n_train)
                .shuffle(len(train_data), reshuffle_each_iteration=shuffle_training).batch(
                    batch_size
                )
            )

        ds_valid = (
            tf.data.Dataset.from_generator(
                validation_data_gen, output_signature=self.eval_signature
            )
            # .take(n_test)
            .batch(test_batch_size)
        )
        ds_test = (
            tf.data.Dataset.from_generator(testing_data_gen, output_signature=self.eval_signature)
            # .take(n_test)
            .batch(test_batch_size)
        )

        return TrainValidTestData(
            ds_train_noshuffle,
            ds_train_valid,
            ds_train,
            ds_valid,
            ds_test,
            train_valid_data,
            train_data,
            valid_data,
            test_data,
            train_valid_idx,
            training_idx,
            valid_idx,
            testset_idx,
            training_valid_data_gen,
            training_data_gen,
            validation_data_gen,
            testing_data_gen,
        )

    def create_kfold_train_valid_test_split(
        self,
        kfolds: int,
        data: TrainValidTestData,
        batch_size=1,
        test_batch_size=1,
        shuffle_training=True,
        seed=1234,
    ):

        # Split up the data into the k folds
        kf = KFold(n_splits=kfolds, shuffle=True, random_state=seed)
        training_idx, valid_idx = list(zip(*list(kf.split(data.training_valid_idx))))

        ds_train: Tuple[tf.data.Dataset, ...] = []
        ds_valid: Tuple[tf.data.Dataset, ...] = []
        train_data: Tuple[list, ...] = []
        valid_data: Tuple[list, ...] = []
        training_gen: Tuple[Callable, ...] = []
        valid_gen: Tuple[Callable, ...] = []

        # remake the data generators
        for j in range(kfolds):
            train_data_fold = [x for i, x in enumerate(self) if i in training_idx[j]]
            valid_data_fold = [x for i, x in enumerate(self) if i in valid_idx[j]]

            def training_data_gen_fold():
                for xy in train_data_fold:
                    yield xy

            def validation_data_gen_fold():
                for xy in valid_data_fold:
                    yield xy

            ds_train_fold = (
                tf.data.Dataset.from_generator(
                    training_data_gen_fold, output_signature=self.eval_signature
                )
                # .take(n_train)
                .shuffle(len(train_data_fold), reshuffle_each_iteration=shuffle_training)
                .batch(batch_size)
                .cache()
            )
            ds_valid_fold = (
                tf.data.Dataset.from_generator(
                    validation_data_gen_fold, output_signature=self.eval_signature
                )
                # .take(n_test)
                .batch(test_batch_size).cache()
            )

            ds_train.append(ds_train_fold)
            ds_valid.append(ds_valid_fold)
            train_data.append(train_data_fold)
            valid_data.append(valid_data_fold)
            training_gen.append(ds_train_fold)
            valid_gen.append(ds_valid_fold)

        return KFoldTrainValidTestData(
            folds=kfolds,
            ds_train_noshuffle=data.ds_train_noshuffle,
            ds_train_valid=data.ds_train_valid,
            ds_train=ds_train,
            ds_valid=ds_valid,
            ds_test=data.ds_test,
            train_valid_data=data.train_valid_data,
            train_data=train_data,
            valid_data=valid_data,
            test_data=data.test_data,
            training_valid_idx=data.training_valid_idx,
            training_idx=training_idx,
            valid_idx=valid_idx,
            testset_idx=data.testset_idx,
            training_valid_gen=data.training_valid_gen,
            training_gen=training_gen,
            valid_gen=valid_gen,
            test_gen=data.test_gen,
        )
