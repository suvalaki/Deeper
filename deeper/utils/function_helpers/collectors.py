import tensorflow as tf


def get_local_tensors(x):
    """Return a dict of all the tensors defined by the current function. This
    function is intended to to be used as the return of internal tfk.Model
    call methods.
    """
    tensors = {k: v for k, v in x.items() if tf.is_tensor(v)}
    return tensors
