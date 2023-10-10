from __future__ import annotations
from typing import List
import tensorflow as tf

@tf.function
def weighted_average(a:List[tf.Tensor], b:tf.Tensor, axis_a=-1, axis_b=-1):
    # Take the weighted average of a along the axis of b
    # a is a set of values grouped by dimension -1
    # b is a set of weights along axis -1
    # this only works when each a has a batch dim and same outer dim as 
    # b axis -1 
    # be must be rank 2 (batch, weights)
    a_stack = tf.stack(a, axis=1)
    broadcast_shape = tf.concat([tf.shape(b), tf.ones_like(tf.shape(a)[2:])], axis=0)
    b_expanded = tf.reshape(b, broadcast_shape)
    result = a_stack * b_expanded  # Broadcasting will take care of the multiplication
    final_result = tf.reduce_sum(result, axis=1)
    return final_result

