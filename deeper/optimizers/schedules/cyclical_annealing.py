import tensorflow as tf

from typeguard import typechecked

from deeper.utils.cooling import linear_cooling
from tensorflow_addons.utils.types import FloatTensorLike


class CyclicalPiecewiseLinearLearningRate(
    tf.keras.optimizers.schedules.LearningRateSchedule
):
    @typechecked
    def __init__(
        self,
        initial_learning_rate: Union[FloatTensorLike, Callable],
        maximal_learning_rate: Union[FloatTensorLike, Callable],
        steps_initial: FloatTensorLike,
        steps_interpolating: FloatTensorLike,
        steps_maximal: FloatTensorLike,
        iterations: int = 0,
        name: str = "CyclicalLearningRate",
        scale_fn: Callable = lambda x: 1.0,
    ):

        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.maximal_learning_rate = maximal_learning_rate
        self.steps_initial = steps_initial
        self.steps_interpolating = steps_interpolating
        self.steps_maximal = steps_maximal
        self.iterations = iterations
        self.name = name
        self.scale_fn = scale_fn

    def __call__(self, step):
        with tf.name_scope(self.name or "CyclicalPiecewiseLinearLearningRate"):
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate"
            )
            dtype = initial_learning_rate.dtype
            maximal_learning_rate = tf.cast(self.maximal_learning_rate, dtype)
            steps_initial = tf.cast(self.steps_initial, dtype)
            steps_interpolating = tf.cast(self.steps_interpolating, dtype)
            steps_maximal = tf.cast(self.steps_maximal, dtype)
            iterations = tf.cast(self.iterations, dtype)

            steps_per_iteration = (
                steps_initial + steps_interpolating + steps_maximal
            )
            ncycle = step / steps_per_iteration
            cycle_step = tf.math.floormod(step, steps_per_iteration)

            in_iterations = tf.math.logical_not(
                tf.math.logical_or(ncycle < iterations, iterations == -1)
            )
            in_initial = cycle_step < steps_initial
            in_interpolation = cycle_step < (
                steps_initial + steps_interpolating
            )

            result = maximal_learning_rate * tf.ones_like(step)
            result_w_interpolation = tf.where(
                in_interpolation,
                linear_cooling(
                    cycle_step - steps_initial,
                    initial_learning_rate,
                    maximal_learning_rate,
                    steps_interpolating,
                ),
                result,
            )
            result_w_initial = tf.where(
                in_initial,
                initial_learning_rate * tf.ones_like(result_w_interpolation),
                result_w_interpolation,
            )
            result_w_in_iterations = tf.where(
                in_iterations,
                maximal_learning_rate * tf.ones_like(result_w_initial),
                result_w_initial,
            )
            return result_w_in_iterations
