import tensorflow as tf
from typing import Dict, List

from deeper.utils.dict import NestedDict, iter_leafs

Tensor = tf.Tensor


def all_same(items):
    return all(x == items[0] for x in items)


def mc_stack_mean_dict(x: List[Dict[str, Tensor]]):
    """Collect a list of dicts by their appropriate keys
    
    Assume equal keys and shapes
    """
    mcx = NestedDict()
    for k, t, v in list(iter_leafs(x[0])):
        mcx[tuple(k)] = tf.reduce_sum(
            tf.stack([NestedDict(y)[tuple(k)] / len(x) for y in x], axis=-1),
            axis=-1,
        )

    return mcx
