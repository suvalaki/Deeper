#### tf.GradientTape
This is a context manager. Tensorflow operations which are executed within the
context manager and are being watched (where watched means to either be a 
`trainable=True` variable or is a tensor `x` that this context managers 
`.watch(x)` method was called on. The gradient tape essentially is used to 
record the gradient of a set of opperations. The tape records the gradient
by calling its' `.gradient()` method. Basically the graadient tape records the
opperations and computes the gradient for that recording. It is possible to 
stop and reset this recording to alter the way gradients are computed 
(`.reset()`). 

Within the 2.0 API gradient tape becomes an essential part 
of training. We need to use the gradient tape to record the
opperations used by our tensorflow `Model` objects and 
functional flows.

#### tf.function
Allows us to decorate arbitrary python code in order to build a computational graph


## Tf.Data

The tensorflow dataset module enable async prefetch of the next batch of
data during training. This reduces the time that the model is bottlenecked
by the cpu. Feeding the data using python results in us processing the 
input data with a single thread; meaning the gpu remains idle while that
processing is taking place. 