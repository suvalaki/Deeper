import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.metrics import adjusted_mutual_info_score
from .utils import chain_call, purity_score


def train(model, X_train, y_train, X_test, y_test, num, epochs, iter, verbose=1):
    
    """
    # Setup datasets
    dataset_train = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .repeat()
        .shuffle()
        .batch(num)
    )
    dataset_test = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .repeat()
        .batch(num)
    )
    """

    print('{:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}'.format(
        'epoch','loss','likelih','z-prior','y-prior', 
        'trAMI', 'teAMI', 'trPUR', 'tePUR'))

    for i in tqdm(range(epochs), position=0):
        for j in tqdm(range(iter), position=1):
            idx_train = np.random.choice(X_train.shape[0],num)
            model.train_step(X_train[idx_train])
        
        if i%verbose==0:
            #Evaluate training metrics
            recon, z_ent, y_ent = chain_call(model.entropy_fn, X_train, num)

            recon = np.array(recon).mean()
            z_ent = np.array(z_ent).mean()
            y_ent = np.array(y_ent).mean()

            loss = - ( recon + z_ent + y_ent )

            idx_tr = chain_call(model.predict, X_train, num).argmax(1)
            idx_te = chain_call(model.predict, X_test, num).argmax(1)
        
            ami_tr = adjusted_mutual_info_score(y_train, idx_tr)
            ami_te = adjusted_mutual_info_score(y_test, idx_te)
            
            purity_train = purity_score(y_train, idx_tr)
            purity_test = purity_score(y_test, idx_te)

            print('{:10d} {:10.5f} {:10.5f} {:10.5f} {:10.5f} '
                         '{:10.5f} {:10.5f} '
                         '{:10.5f} {:10.5f}'.format(
                    i, 
                    loss, recon, z_ent, y_ent,
                    ami_tr, ami_te,
                    purity_train, purity_test))
            
    
