from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Activation, Input, Conv2D, BatchNormalization, Flatten, Dense, Dropout, Reshape, AveragePooling2D, Add
from keras.regularizers import l2

def create_wide_residual_network(input_shape, num_classes, depth, widen_factor=1):
    def batch_norm(): return BatchNormalization(axis=-1, momentum=0.9,
                                                epsilon=1e-5)
    def conv2d(out_channels, kernel_size, strides=1):
      return Conv2D(out_channels, kernel_size, strides=strides, padding='same',
                    kernel_regularizer=l2(0.0005))
    def dense(out_units): return Dense(out_units, kernel_regularizer=l2(0.0005))

    def add_basic_block(x_in, out_channels, strides):
        is_channels_equal = x_in.shape[-1] == out_channels
        bn1 = batch_norm()(x_in)
        bn1 = Activation('relu')(bn1)
        out = conv2d(out_channels, 3, strides)(bn1)
        out = batch_norm()(out)
        out = Activation('relu')(out)
        out = conv2d(out_channels, 3, 1)(out)
        shortcut = x_in if is_channels_equal else conv2d(out_channels, 1, strides)(bn1)
        return Add()([out, shortcut])

    def add_conv_group(x_in, out_channels, n, strides):
        out = add_basic_block(x_in, out_channels, strides)
        for _ in range(1, n):
            out = add_basic_block(out, out_channels, 1)
        return out

    n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
    n = (depth - 4) // 6

    inp = Input(shape=input_shape)
    conv1 = conv2d(n_channels[0], 3)(inp)
    conv2 = add_conv_group(conv1, n_channels[1], n, 1)
    conv3 = add_conv_group(conv2, n_channels[2], n, 2)
    conv4 = add_conv_group(conv3, n_channels[3], n, 2)

    out = batch_norm()(conv4)
    out = Activation('relu')(out)
    out = AveragePooling2D(8)(out)
    out = Flatten()(out)
    out = dense(num_classes)(out)
    out = Activation('softmax')(out)
    return Model(inp, out)