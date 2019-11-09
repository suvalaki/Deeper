# This experiment allows us to understand the behaviour of tensorflow keras 
# models and dictionairy of call variables. We are looking to confirm that 
# in eager mode setting a class attribute as tthe result of tensors will 
# allow us to retrieve their values.

import tensorflow as tf 
import numpy as np

tfk = tf.keras
Model = tf.keras.Model 
Layer = tf.keras.layers.Layer 


class VariableContainer():

    def __init__(self):
        pass 


    def track(self, x, name):
        setattr(self, name, x)
        return x



class PassthroughModel(Model):

    def __init__(self, **kwargs):
        super().__init__(kwargs) 
        self.c = VariableContainer()
        self.dense1 = tfk.layers.Dense(5)
        self.dense2 = tfk.layers.Dense(4)
        self.out = tfk.layers.Dense(1) 


    def call(self, x):

        d1 = self.c.track(self.dense1(x), 'dec1') 
        ed1 = self.c.track(tf.math.exp(d1), 'edec1')
        d2 = self.c.track(self.dense2(ed1), 'dec2')
        ed2 = self.c.track(tf.math.exp(d2), 'ed2')

        out = self.out(ed2)
        
        self.vars = {
            'd1':d1, 'ed1':ed1, 'd2':d2, 'ed2':ed2, 'out':out
        }


    def call2(self, x):

        self.call(x)

        trand = tf.math.exp(self.c.ed2)
        return trand



if __name__=='__main__':

    #setup fake data 
    x = np.array([np.random.uniform(0,1,10) for i in range(100)])

    # Setup model  
    mod = PassthroughModel()

    # run the model 
    mod(x)

    mod.c.dec1

    mod.vars

    mod.call2(x)