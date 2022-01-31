import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import io


class PlotterCallback(tf.keras.callbacks.Callback):
    def __init__(self, logdir, X_train, X_test, y_train, y_test, model):
        self.file_writer = tf.summary.create_file_writer(logdir)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = model

    @staticmethod
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

    @staticmethod
    def image_grid(y):
        """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
        # Create a figure to contain the plot.
        figure = plt.figure(figsize=(10, 10))
        for i in range(100):
            # Start next subplot.
            y_pred_train = y[i]
            img = y_pred_train.reshape((28, 28))
            plt.subplot(10, 10, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(img, cmap=plt.cm.binary)

        return figure

    def on_epoch_end(self, epoch, logs):

        with self.file_writer.as_default():
            # img = y_pred_train.numpy().reshape((-1, 28, 28, 1))
            # tf.summary.image("Training data", img, step=epoch)

            indexes = [
                int(np.nonzero(self.y_train == i)[0][0])
                for i in np.unique(self.y_train)
            ]
            samples = []
            for i in indexes:
                samples.append(self.X_train[i : i + 1])
                for j in range(9):
                    samples.append(
                        self.model(self.X_train[i : (i + 1)]).numpy()
                    )

            tf.summary.image(
                f"generated_data/samples",
                self.plot_to_image(self.image_grid(samples)),
                step=epoch,
            )
            plt.close()

    def on_train_begin(self, args):
        # with self.file_writer.as_default():
        #    img = X_train[0:1].reshape((-1, 28, 28, 1))
        #    tf.summary.image("Real data", img, step=0)
        ...
