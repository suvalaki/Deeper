import tensorflow as tf
import numpy as np

from deeper.utils.metrics import purity_score
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    adjusted_mutual_info_score,
)
from sklearn import metrics


class PurityCallback(tf.keras.callbacks.Callback):
    def __init__(self, tb_callback, X_train, X_test, y_train, y_test):
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

    def on_epoch_end(self, epoch, *args):
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        pur_train = purity_score(self.y_train, y_train_pred)
        pur_test = purity_score(self.y_test, y_test_pred)
        ami_train = adjusted_mutual_info_score(self.y_train, y_train_pred)
        ami_test = adjusted_mutual_info_score(self.y_test, y_test_pred)

        print(
            f"PurityTrain: {pur_train:.3f} PurityTest {pur_test:.3f} AMI Train: {ami_train:.3f} AMI Test: {ami_test:.3f}"
        )
        # print(metrics.cluster.contingency_matrix(self.y_train, y_train_pred))

        with self.tb_callback._train_writer.as_default():
            tf.summary.scalar("cluster_metrics/Purity", pur_train, step=epoch)
            tf.summary.scalar("cluster_metrics/AMI", ami_train, step=epoch)
        with self.tb_callback._val_writer.as_default():
            tf.summary.scalar("cluster_metrics/Purity", pur_test, step=epoch)
            tf.summary.scalar("cluster_metrics/AMI", ami_test, step=epoch)

        for cluster in np.unique(y_train_pred):
            X_cluster = self.X_train[y_train_pred == cluster]
            self._write_cluster_average(
                epoch, X_cluster, f"ClusterAverages/Category_{int(cluster)}"
            )
