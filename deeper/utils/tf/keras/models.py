import tensorflow as tf

from tensorflow.keras.models import Model as tfkModel
from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
from deeper.utils.sampling import mc_stack_mean_dict


class Model(tfkModel):

    intrinsic_metrics = []

    def reset_metrics(self):

        for m in self.intrinsic_metrics:
            m.reset_states()

        super(Model, self).reset_metrics()

    @property
    def metrics(self):
        metrics = super(Model, self).metrics
        metrics += self.intrinsic_metrics
        return metrics

    @property
    def metric_results(self, tonumpy=True):
        return {m.name: m.result() for m in self.metrics}

    @property
    def metric_fns(self):
        class SubSpaceClass:
            ...

        cls = SubSpaceClass()
        for m in self.metrics:
            cls.__setattr__(m.name, m._fn)
        return cls


class GenerativeModel(Model, ABC):
    @abstractmethod
    def sample_one(self, x, training=False):
        """Run the generative model to product a single sample"""

    @tf.function
    def sample(self, samples, x, training=False):
        result = [self.sample_one(x, training) for j in range(samples)]
        return result

    @tf.function
    def monte_carlo_estimate(self, samples, x, y, training=False):
        return mc_stack_mean_dict(self.sample(samples, x, training))