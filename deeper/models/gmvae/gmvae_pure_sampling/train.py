import tensorflow as tf
import numpy as np
from tqdm.autonotebook import tqdm
from sklearn.metrics import adjusted_mutual_info_score
from .utils import chain_call, purity_score

from sklearn.preprocessing import OneHotEncoder

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
    temperature_function=None
):

    #t1 = tqdm(total=epochs, position=0)
    #t2 = tqdm(total=int(X_train.shape[0] // num), position=1, leave=False)

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
            'attch_te',
            "temp"
        )
    )

    for i in range(epochs):

        

        # Setup datasets
        dataset_train = (
            tf.data.Dataset.from_tensor_slices(X_train)
            .repeat(iter_train)
            .shuffle(X_train.shape[0])
            .batch(num)
        )

        if temperature_function is not None:
            temp = temperature_function(model.cooling_distance)

        for x in dataset_train:
            model.train_step(x,samples=samples, batch=batch, temperature=temp)
        
        #for i in range(iter):
        #    idx=np.random.choice(len(X_train), num)
        #    model.train_step(X_train[idx])
        #    t2.update(1)
        #t2.close()

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

            attch_te = (
                np.array(np.unique(idx_te, return_counts=True)[1]).max()
                / len(idx_te)
            )

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
                    temp
                )
            )
            if save is not None:
                model.save_weights(save, save_format='tf')

            model.increment_cooling()

        #t1.update(1)
        #t2.n = 0
        #t2.last_print_n = 0
        #t2.refresh()
    #t1.close()


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

    #t1 = tqdm(total=epochs, position=0)
    #t2 = tqdm(total=int(X_train.shape[0] // num), position=1, leave=False)

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
            'attch_te',
            "nent"
        )
    )


    y_ohe = OneHotEncoder()
    y_train_ohe = np.array(y_ohe.fit_transform(y_train.reshape(-1,1)).todense())
    y_test_ohe = np.array(y_ohe.transform(y_test.reshape(-1,1)).todense())

    for i in range(epochs):

        # Setup datasets
        dataset_train = (
            tf.data.Dataset.from_tensor_slices((X_train, y_train_ohe))
            .repeat(iter_train)
            .shuffle(X_train.shape[0])
            .batch(num)
        )

        for x,y in dataset_train:
            model.pretrain_categories_step(x,y,samples=samples)
        
        #for i in range(iter):
        #    idx=np.random.choice(len(X_train), num)
        #    model.train_step(X_train[idx])
        #    t2.update(1)
        #t2.close()

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

            attch_te = (
                np.array(np.unique(idx_te, return_counts=True)[1]).max()
                / len(idx_te)
            )

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
                    np.nan
                )
            )
            if save is not None:
                model.save_weights(save, save_format='tf')

        #t1.update(1)
        #t2.n = 0
        #t2.last_print_n = 0
        #t2.refresh()
    #t1.close()
