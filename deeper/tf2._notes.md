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