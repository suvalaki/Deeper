import tensorflow as tf
import numpy as np
from tqdm.autonotebook import tqdm
from sklearn.metrics import adjusted_mutual_info_score
from .utils import chain_call, purity_score


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
    beta_z_method=lambda x: 1.0,
    beta_y_method=lambda x: 1.0,
):

    #t1 = tqdm(total=epochs, position=0)
    #t2 = tqdm(total=int(X_train.shape[0] // num), position=1, leave=False)

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
        beta_z = beta_z_method(iter)
        beta_y = beta_y_method(iter)
        dataset_train = (
            tf.data.Dataset.from_tensor_slices(X_train)
            .repeat(iter_train)
            .shuffle(X_train.shape[0])
            .batch(num)
        )

        # Train over the dataset
        for x in dataset_train:
            model.train_step(x,samples=samples, batch=batch, beta_z, beta_y)
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
                    ami_tr, ami_te, purity_train, purity_test, attach_te
                )
            )
            if save is not None:
                model.save_weights(save, save_format='tf')

        #t1.update(1)
        #t2.n = 0
        #t2.last_print_n = 0
        #t2.refresh()
    #t1.close()
