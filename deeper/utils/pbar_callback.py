import tensorflow as tf
from typing import Optional, Sequence
from abc import ABC, abstractmethod


class FilterProgbarLoggerBase(ABC, tf.keras.callbacks.ProgbarLogger):
    def __init__(self, count_mode="samples", stateful_metrics=None):
        super().__init__(count_mode=count_mode, stateful_metrics=stateful_metrics)

    @abstractmethod
    def _filter(self, logname):
        return all(word not in logname for word in self.opt_out)

    def _filter_logs(self, logs):
        return logs and {key: value for key, value in logs.items() if self._filter(key)}

    def on_train_batch_end(self, batch, logs=None):
        super().on_train_batch_end(batch, self._filter_logs(logs))

    def on_test_batch_end(self, batch, logs=None):
        super().on_test_batch_end(batch, self._filter_logs(logs))

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, self._filter_logs(logs))

    def on_test_end(self, logs=None):
        super().on_test_end(self._filter_logs(logs))

    def on_predict_end(self, logs=None):
        super().on_predict_end(self._filter_logs(logs))


class FilterOutProgbarLogger(FilterProgbarLoggerBase):
    def __init__(self, count_mode="steps", stateful_metrics=None, opt_out=[]):
        super().__init__(count_mode=count_mode, stateful_metrics=stateful_metrics)
        self.opt_out = opt_out

    def _filter(self, logname):
        return all(word not in logname for word in self.opt_out)


class FilterInProgbarLogger(FilterProgbarLoggerBase):
    def __init__(self, count_mode="steps", stateful_metrics=None, opt_in=[]):
        super().__init__(count_mode=count_mode, stateful_metrics=stateful_metrics)
        self.opt_in = opt_in

    def _filter(self, logname):
        return any(word in logname for word in self.opt_in)
