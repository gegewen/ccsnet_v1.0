from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import tensorflow as tf

reg_weights = 0.001

class ReflectionPadding3D(Layer):
    def __init__(self, padding=(1, 1, 1), dim_ordering='default', **kwargs):
        super(ReflectionPadding3D, self).__init__(**kwargs)

        if dim_ordering == 'default':
            dim_ordering = K.image_data_format()

        self.padding = padding
        self.dim_ordering = dim_ordering

    def call(self, x, mask=None):
        top_pad = self.padding[0]
        bottom_pad = self.padding[0]
        left_pad = self.padding[1]
        right_pad = self.padding[1]
        front_pad = self.padding[2]
        back_pad = self.padding[2]

        paddings = [[0, 0], [left_pad, right_pad], [top_pad, bottom_pad], [front_pad, back_pad], [0, 0]]

        return tf.pad(x, paddings, mode='REFLECT', name=None)

    def compute_output_shape(self, input_shape):
        if self.dim_ordering == 'tf':
            rows = input_shape[1] + self.padding[0] + self.padding[0] if input_shape[1] is not None else None
            cols = input_shape[2] + self.padding[1] + self.padding[1] if input_shape[2] is not None else None
            dep = input_shape[3] + self.padding[2] + self.padding[2] if input_shape[3] is not None else None

            return (input_shape[0],
                    rows,
                    cols,
                    dep,
                    input_shape[4])
        else:
            raise ValueError('Invalid dim_ordering:', self.dim_ordering)

    def get_config(self):
        config = super(ReflectionPadding3D, self).get_config()
        config.update({'padding': self.padding,
                       'dim_ordering': self.dim_ordering})
        return config