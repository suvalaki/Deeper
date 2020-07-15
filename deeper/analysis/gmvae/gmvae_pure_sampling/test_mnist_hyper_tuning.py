import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import kerastuner as kt 
import tensorflow as tf

from deeper.models.gmvae.gmvae_pure_sampling import model
from sklearn.metrics import adjusted_mutual_info_score


mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
X_train = X_train.reshape(X_train.shape[0], 28 * 28)
X_test = X_test.reshape(X_test.shape[0], 28 * 28)
X_train = (X_train > 0.5).astype(float)
X_test = (X_test > 0.5).astype(float)



def build_model(hp):

    lr = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log', default=1e-3)
    HP_bn = hp.Choice("batch_norm", ["None", "before", "after"], default="None")
    HP_eps = hp.Float("epsilon", 0.0, 1.0, default=1.0)
    HP_zlambda = hp.Float("z_lambda", 1e-5, 10.0, default=1.0)
    HP_ylambda = hp.Float("y_lambda", 1e-5, 10.0, default=1.0)
    HP_connected = hp.Choice("connected_weights", [True, False], default=False)
    HP_dropout = hp.Float("dropout", 0.0, 0.2, default=0.0)
    HP_is_fixed_var = hp.Choice("is_fixed_var", [True, False], default=False)
    with hp.conditional_scope("is_fixed_var", True):
        HP_latent_fixed_var = hp.Float("fixed_var", 1e-12, 1, default=0.1)

    params = {
        "components": 10,
        "input_dimension": X_train.shape[1],
        "embedding_dimensions": [512,512],
        "latent_dimensions": 64,
        "mixture_embedding_dimensions": [512,512],
        "mixture_latent_dimensions": 64,
        "embedding_activations": tf.nn.relu,
        "kind": "binary",
        "learning_rate": 1.0,
        "gradient_clip": None,
        "bn_before": True if HP_bn=='before' else False,
        "bn_after": True if HP_bn=='after' else False,
        "categorical_epsilon": HP_eps,
        "reconstruction_epsilon": HP_eps,
        "latent_epsilon": HP_eps,
        "latent_prior_epsilon": HP_eps,
        "z_kl_lambda": HP_zlambda,
        "c_kl_lambda": HP_ylambda,
        "cat_latent_bias_initializer": None,
        "connected_weights": HP_connected,
        # "optimizer":tf.keras.optimizers.Adam(lr_schedule, epsilon=1e-16),
        "optimizer": tf.keras.optimizers.Adam(lr, epsilon=1e-8),
        "categorical_latent_embedding_dropout": HP_dropout,
        "mixture_latent_mu_embedding_dropout": HP_dropout,
        "mixture_latent_var_embedding_dropout": HP_dropout,
        "mixture_posterior_mu_dropout": HP_dropout,
        "mixture_posterior_var_dropout": HP_dropout,
        "recon_dropouut": HP_dropout,
        'latent_fixed_var': HP_latent_fixed_var,
    }

    m1 = model.Gmvae(**params)
    return m1



class ModelTuner(kt.Tuner):
    
    def run_trial(self, trial, X_train, y_train, X_test, y_test):

        hp = trial.hyperparameters
        HP_samples = hp.Int("samples", 1, 20, default=1)
        epoch_loss_metric = tf.keras.metrics.Mean()

        temp = tf.constant(1.0)

        model = self.hypermodel.build(hp)

        #@tf.function
        def run_train_step(x, y, temp):
            

            model.train_step(
                x,
                samples=HP_samples,
                batch=True,
                beta_z=tf.constant(1.0),
                beta_y=tf.constant(1.0),
                temperature=temp,
            )

            #idx_tr = tf.make_ndarray(model.predict(x)).argmax(1)
            idx_tr = tf.math.argmax(model.predict(x), 1)
            ami_tr = adjusted_mutual_info_score(
                y, idx_tr, average_method="arithmetic"
            )

            epoch_loss_metric.update_state(ami_tr)
            return ami_tr, temp

        num=100
        dataset_train = (
            tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .repeat(1)
            .shuffle(X_train.shape[0], reshuffle_each_iteration=True)
            .batch(num)
        )


        for epoch in range(10):

            self.on_epoch_begin(trial, model, epoch, logs={})
            for batch, (x,y) in enumerate(dataset_train):
                self.on_batch_begin(trial, model, batch, logs={})
                bl, temp = run_train_step(x, y, temp)
                temp -= (0.5 / 25) #/ (50000 / 100)
                batch_loss = float(bl)
                self.on_batch_end(trial, model, batch, logs={'ami': batch_loss})

                if batch % 100 == 0:
                    loss = epoch_loss_metric.result().numpy()
                    print('Batch: {}, Average Ami: {}'.format(batch, loss))
                        
                epoch_loss = epoch_loss_metric.result().numpy()
                self.on_epoch_end(trial, model, epoch, logs={'ami': epoch_loss})
                epoch_loss_metric.reset_states()




if __name__ == "__main__":

    tuner = ModelTuner(
        oracle=kt.oracles.BayesianOptimization(
            objective=kt.Objective("ami", "max"),
            max_trials=2
        ),
        hypermodel=build_model,
        directory="results",
        project_name="gmvae_sampling_mnist_search"
    )

    tuner.search(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
