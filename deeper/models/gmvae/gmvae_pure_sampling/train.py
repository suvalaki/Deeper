import tensorflow as tf
import numpy as np
from tqdm.autonotebook import tqdm
from sklearn.metrics import adjusted_mutual_info_score
from .utils import chain_call, purity_score
import os
import gc
from memory_profiler import profile

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
    temperature_function=None,
    save_results=None,
    beta_z_method=lambda: 1.0,
    beta_y_method=lambda: 1.0,
    tensorboard=None,
):

    # t1 = tqdm(total=epochs, position=0)
    # t2 = tqdm(total=int(X_train.shape[0] // num), position=1, leave=False)

    if tensorboard is not None:
        summary_writer = tf.summary.create_file_writer(tensorboard)

    if save_results is not None:

        header_str = (
            "{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t"
            "{:<10}\t{:<10}\t{:<10}\t{:<10}\t"
            "{:<10}\t{:<10}\t{:<10}"
        ).format(
            "epoch",
            "beta_z",
            "beta_y",
            "loss",
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

        if not os.path.exists(save_results):
            with open(save_results, "w") as results_file:
                results_file.write(header_str)

    tqdm.write(header_str)

    dataset_train = (
        tf.data.Dataset.from_tensor_slices(X_train)
        .repeat(iter_train)
        .shuffle(X_train.shape[0], reshuffle_each_iteration=True)
        .batch(num)
    )

    for i in range(epochs):

        # Setup datasets
        iter = model.cooling_distance
        beta_z = beta_z_method()
        beta_y = beta_y_method()

        if temperature_function is not None:
            temp = temperature_function(iter)

        for x in dataset_train:
            # for id in range(X_train.shape[0] // num):
            # idx = dataset_train[num * id : (num* (id+1))]
            model.train_step(
                x,
                samples=samples,
                batch=batch,
                beta_z=beta_z,
                beta_y=beta_y,
                temperature=temp,
            )

        # for i in range(iter):
        #    idx=np.random.choice(len(X_train), num)
        #    model.train_step(X_train[idx])
        #    t2.update(1)
        # t2.close()

        if i % verbose == 0:
            # Evaluate training metrics
            recon, z_ent, y_ent = chain_call(
                model.entropy_fn, X_train, num_inference
            )
            recon = np.array(recon).mean()
            z_ent = np.array(z_ent).mean()
            y_ent = np.array(y_ent).mean()
            loss = -(recon + z_ent + y_ent)

            idx_tr = chain_call(model.predict, X_train, num_inference).argmax(
                1
            )
            idx_te = chain_call(model.predict, X_test, num_inference).argmax(1)

            ami_tr = adjusted_mutual_info_score(
                y_train, idx_tr, average_method="arithmetic"
            )
            ami_te = adjusted_mutual_info_score(
                y_test, idx_te, average_method="arithmetic"
            )

            attch_te = np.array(
                np.unique(idx_te, return_counts=True)[1]
            ).max() / len(idx_te)

            purity_train = purity_score(y_train, idx_tr)
            purity_test = purity_score(y_test, idx_te)

            value_str = (
                "{:d}\t{:10.5f}\t{:10.5f}\t"
                "{:10.5f}\t{:10.5f}\t{:10.5f}\t{:10.5f}\t"
                "{:10.5f}\t{:10.5f}\t"
                "{:10.5f}\t{:10.5f}\t"
                "{:10.2f}\t{:10.5f}".format(
                    int(model.cooling_distance),
                    beta_z,
                    beta_y,
                    loss,
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

            if save_results is not None and False:
                with open(save_results, "a") as results_file:
                    results_file.write("\n" + value_str)

            tqdm.write(value_str)

            if save is not None and False:
                model.save_weights(save, save_format="tf")

            if tensorboard is not None:
                # plot latent space
                latent_vectors = chain_call(
                    model.latent_sample, X_test, num_inference
                )
                plt_latent_true = plot_latent(latent_vectors, y_test, idx_te)

                with summary_writer.as_default():
                    tf.summary.scalar("beta_z", beta_z, step=iter)
                    tf.summary.scalar("beta_y", beta_y, step=iter)
                    tf.summary.scalar("loss", loss, step=iter)
                    tf.summary.scalar("likelihood", recon, step=iter)
                    tf.summary.scalar("z_prior_entropy", z_ent, step=iter)
                    tf.summary.scalar("y_prior_entropy", y_ent, step=iter)
                    tf.summary.scalar("ami_train", ami_tr, step=iter)
                    tf.summary.scalar("ami_test", ami_te, step=iter)
                    tf.summary.scalar("purity_train", purity_train, step=iter)
                    tf.summary.scalar("purity_test", purity_test, step=iter)
                    tf.summary.scalar(
                        "max_cluster_attachment_test", attch_te, step=iter
                    )
                    tf.summary.scalar("beta_z", beta_z, step=iter)
                    tf.summary.image(
                        "latent", plot_to_image(plt_latent_true), step=iter
                    )

        model.increment_cooling()

        # tf.keras.backend.clear_graph()

        gc.collect()
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
    y_train_ohe = np.array(
        y_ohe.fit_transform(y_train.reshape(-1, 1)).todense()
    )
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
            recon, z_ent, y_ent = chain_call(
                model.entropy_fn,
                X_train,
                num_inference,
                scalar_dict={"temperature": 0.1},
            )

            recon = np.array(recon).mean()
            z_ent = np.array(z_ent).mean()
            y_ent = np.array(y_ent).mean()

            loss = -(recon + z_ent + y_ent)

            idx_tr = chain_call(model.predict, X_train, num_inference).argmax(
                1
            )
            idx_te = chain_call(model.predict, X_test, num_inference).argmax(1)

            ami_tr = adjusted_mutual_info_score(
                y_train, idx_tr, average_method="arithmetic"
            )
            ami_te = adjusted_mutual_info_score(
                y_test, idx_te, average_method="arithmetic"
            )

            attch_te = np.array(
                np.unique(idx_te, return_counts=True)[1]
            ).max() / len(idx_te)

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


def train_even(
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
    temperature_function=None,
    save_results=None,
):

    # t1 = tqdm(total=epochs, position=0)
    # t2 = tqdm(total=int(X_train.shape[0] // num), position=1, leave=False)

    if save_results is not None:

        header_str = (
            "{:<10}\t{:<10}\t{:<10}\t{:<10}\t"
            "{:<10}\t{:<10}\t{:<10}\t{:<10}\t"
            "{:<10}\t{:<10}\t{:<10}"
        ).format(
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
            "temp",
        )

        save_results = os.path.join(os.path.abspath(save_results))

        if not os.path.exists(save_results):
            with open(save_results, "w") as results_file:
                results_file.write(header_str)

    tqdm.write(header_str)

    for i in range(epochs):

        # Setup datasets
        dataset_train = (
            tf.data.Dataset.from_tensor_slices(X_train)
            .repeat(iter_train)
            .shuffle(X_train.shape[0])
            .batch(num)
        )

        for x in dataset_train:
            model.pretrain_step(x, samples=samples, batch=batch)

        # for i in range(iter):
        #    idx=np.random.choice(len(X_train), num)
        #    model.train_step(X_train[idx])
        #    t2.update(1)
        # t2.close()

        if i % verbose == 0:
            # Evaluate training metrics
            recon, z_ent, y_ent = chain_call(
                model.entropy_fn, X_train, num_inference
            )

            recon = np.array(recon).mean()
            z_ent = np.array(z_ent).mean()
            y_ent = np.array(y_ent).mean()

            loss = -(recon + z_ent + y_ent)

            idx_tr = chain_call(model.predict, X_train, num_inference).argmax(
                1
            )
            idx_te = chain_call(model.predict, X_test, num_inference).argmax(1)

            ami_tr = adjusted_mutual_info_score(
                y_train, idx_tr, average_method="arithmetic"
            )
            ami_te = adjusted_mutual_info_score(
                y_test, idx_te, average_method="arithmetic"
            )

            attch_te = np.array(
                np.unique(idx_te, return_counts=True)[1]
            ).max() / len(idx_te)

            purity_train = purity_score(y_train, idx_tr)
            purity_test = purity_score(y_test, idx_te)

            value_str = (
                "{:d}\t{:10.5f}\t{:10.5f}\t{:10.5f}\t{:10.5f}\t"
                "{:10.5f}\t{:10.5f}\t"
                "{:10.5f}\t{:10.5f}\t"
                "{:10.2f}\t{:10.5f}".format(
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
                    0.0,
                )
            )

            if save_results is not None:
                with open(save_results, "a") as results_file:
                    results_file.write("\n" + value_str)

            tqdm.write(value_str)

            if save is not None:
                model.save_weights(save, save_format="tf")

        # t1.update(1)
        # t2.n = 0
        # t2.last_print_n = 0
        # t2.refresh()
    # t1.close()
