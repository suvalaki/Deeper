import itertools
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import io

from sklearn.decomposition import PCA
import seaborn as sns
from deeper.utils.model_mixins import ReconstructionMixin, ClusteringMixin, LatentMixin
from deeper.utils.metrics import purity_score
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    adjusted_mutual_info_score,
)
from sklearn import metrics


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def image_grid(y, nrow=10, ncol=10, figsize=(15, 15), imdims=(28, 28)):
    """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=figsize)
    for i in range(nrow * ncol):
        # Start next subplot.
        y_pred_train = y[i]
        img = y_pred_train.reshape(imdims)
        plt.subplot(nrow, ncol, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img, cmap=plt.cm.binary)

    return figure


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


class ReconstructionImagePlotter(tf.keras.callbacks.Callback):
    def __init__(self, model, tb_callback, X_train, X_test, y_train=None, y_test=None):
        super().__init__()
        self.model = model
        self.tb_callback = tb_callback
        self.X_train = X_train
        self.X_test = X_test

        # assuming y is categorical
        self.y_train = y_train
        self.y_test = y_test

    def _plot_categories(self, epoch):

        n_plots = 100
        if self.y_train is not None:
            # Get the first index from each category in y
            indexes = [int(np.nonzero(self.y_train == i)[0][0]) for i in np.unique(self.y_train)]
        else:
            indexes = [i for i in range(n_plots)]

        samples = []
        for i in indexes:
            samples.append(self.X_train[i : i + 1])
            for j in range(9):
                samples.append(self.model.call_reconstruction(self.X_train[i : (i + 1)]).numpy())

        with self.tb_callback._train_writer.as_default():
            tf.summary.image(
                f"output_reconstruction/samples",
                plot_to_image(image_grid(samples)),
                step=epoch,
            )
        plt.close()

    def on_epoch_end(self, epoch, logs):
        if isinstance(self.model, ReconstructionMixin):
            if self.model.implements_reconstruction:
                self._plot_categories(epoch)


class ClusteringCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, tb_callback, X_train, X_test, y_train, y_test):
        super().__init__()
        self.tb_callback = tb_callback
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def _write_cluster_average(self, epoch, X_cluster, clustername):

        with self.tb_callback._train_writer.as_default():
            img = X_cluster.mean(axis=0).reshape((-1, 28, 28, 1))
            tf.summary.image(clustername, img, step=epoch)

    def _write_clustering_metrics(self, epoch):
        class_names = [str(i) for i in np.unique(self.y_train)]

        X_train = tf.data.Dataset.from_tensor_slices(self.X_train).batch(128)
        X_test = tf.data.Dataset.from_tensor_slices(self.X_test).batch(128)
        y_train_pred = tf.concat(
            [self.model.call_cluster(d) for d in X_train],
            axis=0,
        ).numpy()
        y_test_pred = tf.concat(
            [self.model.call_cluster(d) for d in X_test],
            axis=0,
        ).numpy()

        pur_train = purity_score(self.y_train, y_train_pred)
        pur_test = purity_score(self.y_test, y_test_pred)
        ami_train = adjusted_mutual_info_score(self.y_train, y_train_pred)
        ami_test = adjusted_mutual_info_score(self.y_test, y_test_pred)
        cm_train = metrics.confusion_matrix(self.y_train, y_train_pred)
        cm_test = metrics.confusion_matrix(self.y_test, y_test_pred)

        print(
            f"PurityTrain: {pur_train:.3f} PurityTest {pur_test:.3f} AMI Train: {ami_train:.3f} AMI Test: {ami_test:.3f}"
        )
        # print(metrics.cluster.contingency_matrix(self.y_train, y_train_pred))

        with self.tb_callback._train_writer.as_default():
            tf.summary.scalar("cluster_metrics/Purity", pur_train, step=epoch)
            tf.summary.scalar("cluster_metrics/AMI", ami_train, step=epoch)
            figure = plot_confusion_matrix(cm_train, class_names=class_names)
            cm_image = plot_to_image(figure)
            tf.summary.image("confusion/train", cm_image, step=epoch)

        with self.tb_callback._val_writer.as_default():
            tf.summary.scalar("cluster_metrics/Purity", pur_test, step=epoch)
            tf.summary.scalar("cluster_metrics/AMI", ami_test, step=epoch)
            figure = plot_confusion_matrix(cm_test, class_names=class_names)
            cm_image = plot_to_image(figure)
            tf.summary.image("confusion/test", cm_image, step=epoch)

    def on_epoch_end(self, epoch, *args):

        if isinstance(self.model, ClusteringMixin):
            if self.model.implements_clustering:
                self._write_clustering_metrics(epoch)


class LatentPlotterCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, tb_callback, X_train, X_test, y_train=None, y_test=None):
        super().__init__()
        self.model = model
        self.tb_callback = tb_callback
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_train_str = [str(x) for x in self.y_train] if y_train is not None else None
        self.y_test_str = [str(x) for x in self.y_test] if y_test is not None else None

    def _plot_latent(self, epoch, writer, x, y):
        # Plot the latent space for the training set
        data = tf.data.Dataset.from_tensor_slices(x).batch(128)
        X_test_pca = PCA(2).fit_transform(
            tf.concat(
                [self.model.call_latent(d) for d in data.as_numpy_iterator()],
                axis=0,
            ).numpy()
        )
        plot0 = sns.scatterplot(x=X_test_pca[:, 0], y=X_test_pca[:, 1], hue=y).get_figure()
        with writer.as_default():
            tf.summary.image(
                f"generated_data/latent_sample",
                plot_to_image(plot0),
                step=epoch,
            )
        plt.close()

        with writer.as_default():
            # Plot the kdeplot of the latent set
            plot1 = sns.kdeplot(x=X_test_pca[:, 0], y=X_test_pca[:, 1], hue=y).get_figure()

            tf.summary.image(
                f"generated_data/latent_kde",
                plot_to_image(plot1),
                step=epoch,
            )
        plt.close()

    def on_epoch_end(self, epoch, logs):

        if isinstance(self.model, LatentMixin):
            if self.model.implements_latent:

                for writer, x, y in zip(
                    [
                        self.tb_callback._train_writer,
                        self.tb_callback._val_writer,
                    ],
                    [self.X_train, self.X_test],
                    [self.y_train_str, self.y_test_str],
                ):
                    self._plot_latent(epoch, writer, x, y)
