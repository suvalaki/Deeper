import tensorflow as tf
import numpy as np
from tqdm.autonotebook import tqdm
from sklearn.metrics import adjusted_mutual_info_score
from .utils import chain_call, purity_score
import os

from sklearn.preprocessing import OneHotEncoder

from deeper.utils.tensorboard import plot_to_image
from deeper.models.gmvae.utils import plot_latent


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
    temperature_function=lambda: 0.5,
    save_results=None,
    beta_z_method=lambda: 1.0,
    beta_y_method=lambda: 1.0,
    beta_d_method=lambda: 1.0,
    tensorboard="./logs",
):

    # t1 = tqdm(total=epochs, position=0)
    # t2 = tqdm(total=int(X_train.shape[0] // num), position=1, leave=False)

    summary_writer = tf.summary.create_file_writer(tensorboard)

    if save_results is not None:

        header_str = (
            "{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t"
            "{:<10}\t{:<10}\t{:<10}\t{:<10}\t"
            "{:<10}\t{:<10}\t{:<10}"
        ).format(
            "epoch",
            "beta_z",
            "beta_y",
            "loss",
            "gan_ent",
            "likelih",
            "z-prior",
            "y-prior",
            "trAMI",
            "teAMI",
            "trPUR",
            "tePUR",
            "attch_te",
            "temp",
        )

        save_results = os.path.join(os.path.abspath(save_results))

    for i in range(epochs):

        # Setup datasets
        dataset_train = (
            tf.data.Dataset.from_tensor_slices(X_train)
            .repeat(iter_train)
            .shuffle(X_train.shape[0])
            .batch(num)
        )

        iter = model.cooling_distance
        beta_z = beta_z_method()
        beta_y = beta_y_method()
        beta_d = beta_d_method()
        if temperature_function is not None:
            temp = temperature_function(iter)
        else:
            temp = 1.0

        for x in dataset_train:
            model.train_step(
                x,
                samples=samples,
                temperature=temp,
                beta_z=beta_z,
                beta_y=beta_y,
                beta_d=beta_d,
                gradient_clip=model.gradient_clip,
            )

        # for i in range(iter):
        #    idx=np.random.choice(len(X_train), num)
        #    model.train_step(X_train[idx])
        #    t2.update(1)
        # t2.close()

        if i % verbose == 0:
            # Evaluate training metrics
            recon, z_ent, y_ent, desc_ent = chain_call(model.entropy_fn, X_train, num_inference)

            recon = np.array(recon).mean()
            z_ent = np.array(z_ent).mean()
            y_ent = np.array(y_ent).mean()
            d_ent = np.array(desc_ent).mean()

            loss = -(recon + z_ent + y_ent)

            idx_tr = chain_call(model.predict, X_train, num_inference).argmax(1)
            idx_te = chain_call(model.predict, X_test, num_inference).argmax(1)

            ami_tr = adjusted_mutual_info_score(y_train, idx_tr, average_method="arithmetic")
            ami_te = adjusted_mutual_info_score(y_test, idx_te, average_method="arithmetic")

            attch_te = np.array(np.unique(idx_te, return_counts=True)[1]).max() / len(idx_te)

            purity_train = purity_score(y_train, idx_tr)
            purity_test = purity_score(y_test, idx_te)
            value_str = (
                "{:d}\t{:10.5f}\t{:10.5f}\t{:10.5f}\t"
                "{:10.5f}\t{:10.5f}\t{:10.5f}\t{:10.5f}\t"
                "{:10.5f}\t{:10.5f}\t"
                "{:10.5f}\t{:10.5f}\t"
                "{:10.2f}\t{:10.5f}".format(
                    iter,
                    beta_z,
                    beta_y,
                    loss,
                    d_ent,
                    recon,
                    z_ent,
                    y_ent,
                    ami_tr,
                    ami_te,
                    purity_train,
                    purity_test,
                    attch_te,
                    temp,
                )
            )

            if save_results is not None:
                with open(save_results, "a") as results_file:
                    results_file.write("\n" + value_str)

            tqdm.write(value_str)

            model.increment_cooling()

            # plot latent space
            latent_vectors = chain_call(model.latent_sample, X_test, num_inference)
            plt_latent_true = plot_latent(latent_vectors, y_test, idx_te)

            with summary_writer.as_default():
                tf.summary.scalar("beta_z", beta_z, step=iter)
                tf.summary.scalar("beta_y", beta_y, step=iter)
                tf.summary.scalar("loss", loss, step=iter)
                tf.summary.scalar("gan_entropy", d_ent, step=iter)
                tf.summary.scalar("likelihood", recon, step=iter)
                tf.summary.scalar("z_prior_entropy", z_ent, step=iter)
                tf.summary.scalar("y_prior_entropy", y_ent, step=iter)
                tf.summary.scalar("ami_train", ami_tr, step=iter)
                tf.summary.scalar("ami_test", ami_te, step=iter)
                tf.summary.scalar("purity_train", purity_train, step=iter)
                tf.summary.scalar("purity_test", purity_test, step=iter)
                tf.summary.scalar("max_cluster_attachment_test", attch_te, step=iter)
                tf.summary.scalar("beta_z", beta_z, step=iter)
                tf.summary.image("latent", plot_to_image(plt_latent_true), step=iter)

        # t1.update(1)
        # t2.n = 0
        # t2.last_print_n = 0
        # t2.refresh()
    # t1.close()


def pretrain_with_clusters(
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
):

    # t1 = tqdm(total=epochs, position=0)
    # t2 = tqdm(total=int(X_train.shape[0] // num), position=1, leave=False)

    tqdm.write(
        "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
            "epoch",
            "loss",
            "likelih",
            "z-prior",
            "y-prior",
            "trAMI",
            "teAMI",
            "trPUR",
            "tePUR",
            "attch_te",
            "nent",
        )
    )

    y_ohe = OneHotEncoder()
    y_train_ohe = np.array(y_ohe.fit_transform(y_train.reshape(-1, 1)).todense())
    y_test_ohe = np.array(y_ohe.transform(y_test.reshape(-1, 1)).todense())

    for i in range(epochs):

        # Setup datasets
        dataset_train = (
            tf.data.Dataset.from_tensor_slices((X_train, y_train_ohe))
            .repeat(iter_train)
            .shuffle(X_train.shape[0])
            .batch(num)
        )

        for x, y in dataset_train:
            model.pretrain_categories_step(x, y, samples=samples)

        # for i in range(iter):
        #    idx=np.random.choice(len(X_train), num)
        #    model.train_step(X_train[idx])
        #    t2.update(1)
        # t2.close()

        if i % verbose == 0:
            # Evaluate training metrics
            recon, z_ent, y_ent = chain_call(model.entropy_fn, X_train, num_inference)

            recon = np.array(recon).mean()
            z_ent = np.array(z_ent).mean()
            y_ent = np.array(y_ent).mean()

            loss = -(recon + z_ent + y_ent)

            idx_tr = chain_call(model.predict, X_train, num_inference).argmax(1)
            idx_te = chain_call(model.predict, X_test, num_inference).argmax(1)

            ami_tr = adjusted_mutual_info_score(y_train, idx_tr, average_method="arithmetic")
            ami_te = adjusted_mutual_info_score(y_test, idx_te, average_method="arithmetic")

            attch_te = np.array(np.unique(idx_te, return_counts=True)[1]).max() / len(idx_te)

            purity_train = purity_score(y_train, idx_tr)
            purity_test = purity_score(y_test, idx_te)

            tqdm.write(
                "{:10d} {:10.5f} {:10.5f} {:10.5f} {:10.5f} "
                "{:10.5f} {:10.5f} "
                "{:10.5f} {:10.5f} "
                "{:10.2f} {:10.5f}".format(
                    i,
                    loss,
                    recon,
                    z_ent,
                    y_ent,
                    ami_tr,
                    ami_te,
                    purity_train,
                    purity_test,
                    attch_te,
                    np.nan,
                )
            )
            if save is not None:
                model.save_weights(save, save_format="tf")

        # t1.update(1)
        # t2.n = 0
        # t2.last_print_n = 0
        # t2.refresh()
    # t1.close()
