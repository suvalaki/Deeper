
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
import pandas as pd
import io


def plot_latent(latent_vectors, y_test, y_pred):
    #pca = PCA(2)
    pca = TSNE(2)
    X_pca = pca.fit_transform(latent_vectors)
    kmeans = GaussianMixture(10, tol=1e-6, max_iter = 1000)
    pred = kmeans.fit_predict(X_pca)

    df_latent = pd.DataFrame({
        'x1':X_pca[:,0], 
        'x2':X_pca[:,1], 
        'cat':y_test,#['pred_{}'.format(i) for i in y_test],
        'kmeans':y_pred#['pred_{}'.format(i) for i in pred]
    })

    #fig, ax = plt.subplots()

    #km_pur = purity_score()

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(20,10))

    #true_scatter = sns.scatterplot(data=df_latent,x='x1',y='x2',hue='cat', ax=ax)
    ax1.scatter(
        df_latent.x1, df_latent.x2, c=df_latent.cat, cmap='viridis'
    )
    ax1.set_title('True Labels. VAE purity ()')

    #fig2, ax2 = plt.subplots()
    #pred_scatter = sns.scatterplot(data=df_latent,x='x1',y='x2',hue='kmeans', ax=ax2)
    ax2.scatter(
        df_latent.x1, df_latent.x2, c=df_latent.kmeans, cmap='viridis'
    )
    ax2.set_title('Latent Clustering Labels. Purity (pred)')

    return f
    #return true_scatter, pred_scatter