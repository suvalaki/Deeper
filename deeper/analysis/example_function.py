import tensorflow as tf


def f(x, y):
    return tf.reduce_mean(tf.multiply(x ** 2, 3) + y)


g = tf.function(f)

x = tf.constant([[2.0, 3.0]])
y = tf.constant([[3.0, -2.0]])

# `f` and `g` will return the same value, but `g` will be executed as a
# TensorFlow graph.
assert f(x, y).numpy() == g(x, y).numpy()

# Tensors and tf.Variables used by the Python function are captured in the
# graph.
@tf.function
def h():
    return f(x, y)


assert (h().numpy() == f(x, y).numpy()).all()


# Data-dependent control flow is also captured in the graph. Supported
# control flow statements include `if`, `for`, `while`, `break`, `continue`,
# `return`.
@tf.function
def g(x):
    if tf.reduce_sum(x) > 0:
        return x * x
    else:
        return -x // 2


c = tf.Variable(0)


@tf.function
def f(x):
    c.assign_add(1)
    return x + tf.compat.v1.to_float(c)


assert c == 0.0
assert f(1.0) == 2.0
assert int(c) == 1
assert f(1.0) == 3.0
assert int(c) == 2


class Dense(object):
    def __init__(self):
        self.W = tf.Variable(tf.compat.v1.glorot_uniform_initializer()((10, 10)))
        self.b = tf.Variable(tf.zeros(10))

    @tf.function
    def compute(self, x):
        return tf.matmul(x, self.W) + self.b


d1 = Dense()
d2 = Dense()
x = tf.random.uniform((10, 10))
# d1 and d2 are using distinct variables
assert not (d1.compute(x).numpy() == d2.compute(x).numpy()).all()


# KERAS EXAMPLE


class MyModel(tf.keras.Model):
    def __init__(self, keep_probability=0.2):
        tf.keras.Model.__init__(self)
        self.dense1 = tf.keras.layers.Dense(4)
        self.dense2 = tf.keras.layers.Dense(5)
        self.keep_probability = keep_probability

    @tf.function
    def call(self, inputs, training=True):
        y = self.dense2(self.dense1(inputs))
        if training:
            return tf.nn.dropout(y, self.keep_probability)
        else:
            return y


x = tf.random.uniform((100000, 1000))

model = MyModel()
model(x, training=True)  # executes a graph, with dropout
model(x, training=False)  # executes a graph, without dropout
