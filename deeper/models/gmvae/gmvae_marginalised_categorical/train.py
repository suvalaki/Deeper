import tensorflow as tf
import numpy as np
from tqdm.autonotebook import tqdm
from sklearn.metrics import adjusted_mutual_info_score
from .utils import chain_call, purity_score
from deeper.utils.cooling import CyclicCoolingRegime
from matplotlib import pyplot as plt 
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
import pandas as pd
import io


def plot_latent(latent_vectors, y_test):
    pca = PCA(2)
    #pca = TSNE(2)
    X_pca = pca.fit_transform(latent_vectors)
    kmeans = GaussianMixture(10, tol=1e-6, max_iter = 1000)
    pred = kmeans.fit_predict(X_pca)

    df_latent = pd.DataFrame({
        'x1':X_pca[:,0], 
        'x2':X_pca[:,1], 
        'cat':y_test,#['pred_{}'.format(i) for i in y_test],
        'kmeans':pred#['pred_{}'.format(i) for i in pred]
    })

    #fig, ax = plt.subplots()

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,20))

    #true_scatter = sns.scatterplot(data=df_latent,x='x1',y='x2',hue='cat', ax=ax)
    ax1.scatter(
        df_latent.x1, df_latent.x2, c=df_latent.cat, cmap='viridis'
    )

    #fig2, ax2 = plt.subplots()
    #pred_scatter = sns.scatterplot(data=df_latent,x='x1',y='x2',hue='kmeans', ax=ax2)
    ax2.scatter(
        df_latent.x1, df_latent.x2, c=df_latent.cat, cmap='viridis'
    )

    return f
    #return true_scatter, pred_scatter


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image



def train(
    model, 
    X_train, 
    y_train, 
    X_test, 
    y_test, 
    num, 
    samples,
    epochs, 
    iter_train, 
    num_inference, 
    batch=False,
    verbose=1, 
    save=None,
    beta_z_method=lambda: 1.0,
    beta_y_method=lambda: 1.0,
    tensorboard='./logs',
):

    #t1 = tqdm(total=epochs, position=0)
    #t2 = tqdm(total=int(X_train.shape[0] // num), position=1, leave=False)

    summary_writer = tf.summary.create_file_writer(tensorboard)

    tqdm.write(
        "{:>10} {:>10} {:>10} "
        "{:>10} {:>10} {:>10} {:>10} "
        "{:>10} {:>10} {:>10} {:>10} {:>10}".format(
            "epoch", "beta_z", "beta_y",
            "loss","likelih","z-prior","y-prior",
            "trAMI","teAMI","trPUR","tePUR","te_ATCH"
        )
    )

    for i in range(epochs):

        # Setup datasets
        iter = model.cooling_distance
        beta_z = beta_z_method()
        beta_y = beta_y_method()
        dataset_train = (
            tf.data.Dataset.from_tensor_slices(X_train)
            .repeat(iter_train)
            .shuffle(X_train.shape[0])
            .batch(num)
        )

        # Train over the dataset
        for x in dataset_train:
            model.train_step(
                x, 
                samples=samples, 
                batch=batch, 
                beta_z=beta_z, 
                beta_y=beta_y
            )
        model.increment_cooling()
        

        if i % verbose == 0:
            # Evaluate training metrics
            recon, z_ent, y_ent = chain_call(model.entropy_fn, X_train, num_inference)

            recon = np.array(recon).mean()
            z_ent = np.array(z_ent).mean()
            y_ent = np.array(y_ent).mean()

            loss = -(recon + z_ent + y_ent)

            idx_tr = chain_call(model.predict, X_train, num_inference).argmax(1)
            idx_te = chain_call(model.predict, X_test, num_inference).argmax(1)

            ami_tr = adjusted_mutual_info_score(
                y_train, idx_tr, average_method="arithmetic"
            )
            ami_te = adjusted_mutual_info_score(
                y_test, idx_te, average_method="arithmetic"
            )

            purity_train = purity_score(y_train, idx_tr)
            purity_test = purity_score(y_test, idx_te)

            attch_te = (
                np.array(np.unique(idx_te, return_counts=True)[1]).max()
                / len(idx_te)
            )

            tqdm.write(
                "{:10d} {:10.5f} {:10.5f} "
                "{:10.5f} {:10.5f} {:10.5f} {:10.5f} "
                "{:10.5f} {:10.5f} {:10.5f} {:10.5f} {:10.5f} "
                .format(
                    iter, beta_z, beta_y,
                    loss, recon, z_ent, y_ent,
                    ami_tr, ami_te, purity_train, purity_test, attch_te
                )
            )
            if save is not None:
                model.save_weights(save, save_format='tf')

            #plot latent space 
            latent_vectors = chain_call(
                model.latent_sample, 
                X_test, 
                num_inference
            )
            plt_latent_true  = plot_latent(
                latent_vectors, 
                y_test
            )

            with summary_writer.as_default():
                tf.summary.scalar('beta_z', beta_z, step=iter)
                tf.summary.scalar('beta_y', beta_y, step=iter)
                tf.summary.scalar('loss', loss, step=iter)
                tf.summary.scalar('likelihood', recon, step=iter)
                tf.summary.scalar('z_prior_entropy', z_ent, step=iter)
                tf.summary.scalar('y_prior_entropy', y_ent, step=iter)
                tf.summary.scalar('ami_train', ami_tr, step=iter)
                tf.summary.scalar('ami_test', ami_te, step=iter)
                tf.summary.scalar('purity_train', purity_train, step=iter)
                tf.summary.scalar('purity_test', purity_test, step=iter)
                tf.summary.scalar('max_cluster_attachment_test', attch_te, step=iter)
                tf.summary.scalar('beta_z', beta_z, step=iter)
                tf.summary.image(
                    "latent", plot_to_image(plt_latent_true), step=iter
                )

        #t1.update(1)
        #t2.n = 0
        #t2.last_print_n = 0
        #t2.refresh()
    #t1.close()
