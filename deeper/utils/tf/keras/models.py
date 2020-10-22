import tensorflow as tf 

from tensorflow.keras.models import Model as tfkModel 


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
    def metric_results(self, numpy=True):
        if numpy:
            return {m.name: m.result().numpy() for m in self.metrics}
        else:
            return {m.name: m.result() for m in self.metrics}


    @property
    def metric_fns(self):
        class SubSpaceClass:
            ...
        cls = SubSpaceClass()
        for m in self.metrics:
            cls.__setattr__(m.name, m._fn)
        return cls