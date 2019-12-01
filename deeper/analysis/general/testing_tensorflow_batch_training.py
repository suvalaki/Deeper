import tensorflow as tf
import numpy as np

# Question of Interest. When batch size = None Does tensorflow automatically
# create a single gradient vector. Instead of a gradient matrix. Otherwise
# stated; does tensorflow apply a minibatch training update or a single data
# update on computation of the gradient.

tf.enable_eager_execution()

#%% Validation for a single input
# Create a variable.
w = tf.Variable([2.0], name="weight", shape=[1], dtype=tf.float64)
# Use the variable in the graph like any Tensor.
@tf.function
def graph(x):
    y = tf.multiply(w, x)
    return y


@tf.function
def graph_loss(x, y):
    mse = tf.reduce_mean(tf.square(graph(x) - y), -1)
    return mse


# Setup the gradient operation
@tf.function
def graph_gradients(x, y):
    with tf.GradientTape() as tape:
        loss = graph_loss(x, y)
    grad = tape.gradient(loss, sources=[w])
    return grad


# Test the result
y_train = tf.constant([7.0], dtype=tf.float64, name="y_train")
x_train = tf.constant([1.0], dtype=tf.float64, name="x_train")
grad = graph_gradients(x_train, y_train)
print(grad[0])

#%% Verify with Multiple input dim
x_train_batch = tf.constant(
    np.array([[1.0], [2.0]]), dtype=tf.float64, name="x_train_batch"
)
y_train_batch = tf.constant(
    np.array([[72.0], [42.0]]), dtype=tf.float64, name="x_train_batch"
)
result = graph(x_train_batch)
grad_batch = graph_gradients(x_train_batch, y_train_batch)
print(grad_batch[0])

#%% Finally inspect the gradients to check they are the same shape
assert grad[0].shape == grad_batch[0].shape, "Gradient shape changed"

# Batch training still only computes a single gradients for the variable

#%% Are the gradients computed the sum over all the input gradients?
grad_val = 0
input_list = ((1.0, 72.0), (2.0, 42.0))
for in_val, out_val in input_list:
    grad_val += graph_gradients(in_val, out_val)[0]

assert (
    grad_batch[0].numpy() == grad_val.numpy()
), "the gradients are NOT being summed together"

# Gradients over a batch ARE being summed together

# Hence we can understand that to feed multiple values into the tensorflow
# graph will still only result in a single backwards pass. Gradients are taken
# as the sum over all of the gradients in that single backwards pass.
# This is NOT the same as averaging the gradients.
#
# One Reason this may be inportant is that bigger the batch the bigger will be
# the gradient at the end. And so if we are clipping the gradients we will
# clip depending on the batch size. Bigger batches will destructively clip

#%% Verify that the average over the entire batch can be found by taking the
# mean of the input loss functions


@tf.function
def graph_average_loss_gradients(x, y):
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(graph_loss(x, y))
    grad = tape.gradient(loss, sources=[w])
    return grad


grad_ave_batch = graph_average_loss_gradients(x_train_batch, y_train_batch)
assert grad_ave_batch[0].numpy() == grad_val.numpy() / len(input_list), (
    "Placing  reduce Mean operator at the end does NOT result in \n"
    "averaged gradients"
)

# Reducing the mean of the last layer DOES in fact average the gradients
# It is therefore NOT necessary to average the gradients themselves. The same
# Can be achieved with a reduce_mean applied to the loss fn.

#%%
