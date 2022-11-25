import tensorflow as tf


class RMSE(tf.keras.metrics.Metric):
    def __init__(self, name="RMSE", **kwargs) -> None:
        super(RMSE, self).__init__(name=name, **kwargs)
        self.rmse_sum = self.add_weight(name="rmse_sm", initializer="zeros", **kwargs)
        self.total_n_samples = self.add_weight(name="total_n_samples", initializer="zeros", **kwargs)

    def update_state(self, y_true, y_pred, **kwargs):
        subtraction = tf.subtract(y_true, y_pred)
        squared = tf.square(subtraction)
        meaned = tf.reduce_sum(squared)
        self.rmse_sum.assign_add(meaned)
        print(self.rmse_sum)
        self.total_n_samples.assign_add(tf.shape(y_pred)[0])
        print(self.total_n_samples)
    
    def result(self):
        return tf.sqrt(tf.divide(self.rmse_sum, self.total_n_samples))

    def reset_state(self):
        self.rmse_sum.assign(0.)
        self.total_n_samples.assign(0)
