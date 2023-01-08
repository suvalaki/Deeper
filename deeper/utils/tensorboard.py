import io
from matplotlib import pyplot as plt
import tensorflow as tf


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


def convert_tb_data(root_dir, sort_by=None):
    """Convert local TensorBoard data into Pandas DataFrame.

    Function takes the root directory path and recursively parses
    all events data.
    If the `sort_by` value is provided then it will use that column
    to sort values; typically `wall_time` or `step`.

    *Note* that the whole data is converted into a DataFrame.
    Depending on the data size this might take a while. If it takes
    too long then narrow it to some sub-directories.

    Paramters:
        root_dir: (str) path to root dir with tensorboard data.
        sort_by: (optional str) column name to sort by.

    Returns:
        pandas.DataFrame with [wall_time, name, step, value] columns.

    """
    import os
    import pandas as pd
    from tensorflow.python.summary.summary_iterator import summary_iterator
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

    def convert_tfevent(filepath):

        acc = EventFileLoader(filepath)
        acc.Reload()
        import pdb

        pdb.set_trace()
        return dict()

        # return pd.DataFrame(
        #    [parse_tfevent(e, filepath) for e in summary_iterator(filepath) if len(e.summary.value)]
        # )

    def parse_tfevent(tfevent, filepath):

        return dict(
            path=filepath,
            wall_time=tfevent.wall_time,
            metric=tfevent.summary.value[0].tag,
            step=tfevent.step,
            value=float(tfevent.summary.value[0].simple_value),
        )

    columns_order = ["path", "wall_time", "metric", "step", "value"]

    out = []
    for (root, _, filenames) in os.walk(root_dir):
        for filename in filenames:

            if "events.out.tfevents" not in filename:
                continue
            file_full_path = os.path.join(root, filename)
            out.append(convert_tfevent(file_full_path))

    # Concatenate (and sort) all partial individual dataframes
    all_df = pd.concat(out)[columns_order]
    if sort_by is not None:
        all_df = all_df.sort_values(sort_by)

    return all_df.reset_index(drop=True)


# if __name__ == "__main__":
#     dir_path = "/home/kretyn/projects/ai-traineree/runs/"
#     exp_name = "CartPole-v1_2021-01-26_11:02"
#     df = convert_tb_data(f"{dir_path}/{exp_name}")

#     print(df.head())