from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import add
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv3D, UpSampling3D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import regularizers
import tensorflow as tf

reg_weights = 0.001

def bn_relu():
    def bn_relu_func(x):
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    return bn_relu_func


def conv_bn_relu(nb_filter, nb_row, nb_col, nb_dep, stride):
    def conv_func(x):
        x = Conv3D(nb_filter, (nb_row, nb_col, nb_dep), strides=stride, padding='same', kernel_regularizer=regularizers.l2(reg_weights))(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    return conv_func


def res_conv(nb_filter, nb_row, nb_col, nb_dep, stride=(1, 1, 1)):
    def _res_func(x):
        identity = x

        a = Conv3D(nb_filter, (nb_row, nb_col, nb_dep), strides=stride, padding='same', kernel_regularizer=regularizers.l2(reg_weights))(x)
        a = BatchNormalization()(a)
        a = Activation("relu")(a)
        a = Conv3D(nb_filter, (nb_row, nb_col, nb_dep), strides=stride, padding='same', kernel_regularizer=regularizers.l2(reg_weights))(a)
        y = BatchNormalization()(a)

        return add([identity, y])

    return _res_func


def dconv_bn_nolinear(nb_filter, nb_row, nb_col, nb_dep, stride=(2, 2, 2), activation="relu"):
    def _dconv_bn(x):
        x = UpSampling3D(size=stride)(x)
        x = ReflectionPadding3D(padding=(int(nb_row/2), int(nb_col/2), int(nb_dep/2)))(x)
        x = Conv3D(nb_filter, (nb_row, nb_col, nb_dep), padding='valid', kernel_regularizer=regularizers.l2(reg_weights))(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        return x

    return _dconv_bn


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
        
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Flatten, Dense, Lambda, Reshape, concatenate
from tensorflow.keras.models import Model

def create_vae(input_shape):
    # Encoder
    input = Input(shape=input_shape, name='image')

    enc1_conv = Conv3D(32, (3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=regularizers.l2(reg_weights))(input)
    enc1_bn_relu = bn_relu()(enc1_conv)
    
    enc2_conv = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same', kernel_regularizer=regularizers.l2(reg_weights))(enc1_bn_relu)
    enc2_bn_relu = bn_relu()(enc2_conv)
    
    enc3_conv = Conv3D(128, (3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=regularizers.l2(reg_weights))(enc2_bn_relu)
    enc3_bn_relu = bn_relu()(enc3_conv)
    
    enc4_conv = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same', kernel_regularizer=regularizers.l2(reg_weights))(enc3_bn_relu)
    enc4_bn_relu = bn_relu()(enc4_conv)
    
    enc5_conv = Conv3D(256, (3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=regularizers.l2(reg_weights))(enc4_bn_relu)
    enc5_bn_relu = bn_relu()(enc5_conv)
    
    enc6_conv = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same', kernel_regularizer=regularizers.l2(reg_weights))(enc5_bn_relu)
    enc6_bn_relu = bn_relu()(enc6_conv)


    x0 = res_conv(256, 3, 3, 3)(enc6_bn_relu)
    x1 = res_conv(256, 3, 3, 3)(x0)
    x2 = res_conv(256, 3, 3, 3)(x1)
    x3 = res_conv(256, 3, 3, 3)(x2)
    x4 = res_conv(256, 3, 3, 3)(x3)
    x5 = res_conv(256, 3, 3, 3)(x4)
    dec6 = res_conv(256, 3, 3, 3)(x5)

    dec5 = dconv_bn_nolinear(256, 3, 3, 3, stride=(1, 1, 1))(dec6)
    dec4 = dconv_bn_nolinear(256, 3, 3, 3, stride=(2, 2, 2))(dec5)
    dec3 = dconv_bn_nolinear(128, 3, 3, 3, stride=(1, 1, 1))(dec4)
    dec2 = dconv_bn_nolinear(128, 3, 3, 3, stride=(2, 2, 2))(dec3)
    dec1 = dconv_bn_nolinear(64, 3, 3, 3, stride=(1, 1, 1))(dec2)
    dec0 = dconv_bn_nolinear(32, 3, 3, 3, stride=(2, 2, 2))(dec1)
    
    output = [Conv3D(1, (3, 3, 3), padding='same', activation=None)(dec0)]

    # Full net
    vae_model = Model(input, output)

    return vae_model