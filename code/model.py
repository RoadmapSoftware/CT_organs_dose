from keras.engine import Model
from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D, concatenate
from keras.layers import Conv3DTranspose


def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_last", batch_normalization=False):
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters, batch_normalization=batch_normalization)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters, batch_normalization=batch_normalization)
    return convolution2

def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=LeakyReLU,
                             padding='same', strides=(1, 1, 1), instance_normalization=True, name="conv"):
    if name == "conv":
        layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    elif name == "deconv":
        layer = Conv3DTranspose(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=-1)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=-1)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)

def create_residual_modual(input_layer, n_level_filters, batch_normalization=False):
    context_layer = create_context_module(input_layer, n_level_filters, batch_normalization=batch_normalization)
    add_layer = Add()([input_layer, context_layer])
    residual_layer = create_convolution_block(add_layer, n_level_filters, batch_normalization=batch_normalization)
    return residual_layer

def creat_segmentation_modual(input_layer, n_filters, batch_normalization=False):
    convolution1 = create_convolution_block(input_layer, n_filters, kernel=(1, 1, 1), batch_normalization=batch_normalization)
    convolution2 = create_convolution_block(convolution1, n_filters, batch_normalization=batch_normalization)
    return convolution2

def Unet3D_model(input_shape=(128, 128, 128, 1), n_base_filters=32, n_classes=2, batch_normalization=False, activation_name="softmax"):

    inputs = Input(input_shape)
    
    # encoder
    c1 = create_convolution_block(inputs, n_base_filters, batch_normalization=batch_normalization)
    r1 = create_residual_modual(c1, n_base_filters)
    
    c2 = create_convolution_block(r1, n_base_filters*2, strides=(2, 2, 2), batch_normalization=batch_normalization)
    r2 = create_residual_modual(c2, n_base_filters*2) 
    
    c3 = create_convolution_block(r2, n_base_filters*4, strides=(2, 2, 2), batch_normalization=batch_normalization)
    r3 = create_residual_modual(c3, n_base_filters*4) 
    
    c4 = create_convolution_block(r3, n_base_filters*8, strides=(2, 2, 2), batch_normalization=batch_normalization)
    r4 = create_residual_modual(c4, n_base_filters*8) 
    
    c5 = create_convolution_block(r4, n_base_filters*16, strides=(2, 2, 2), batch_normalization=batch_normalization)
    r5 = create_residual_modual(c5, n_base_filters*16) 
    
    # decoder
    d1 = create_convolution_block(r5, n_base_filters*8, strides=(2, 2, 2), name="deconv", batch_normalization=batch_normalization)
    m1 = concatenate([r4, d1], axis=-1)
    s1 = creat_segmentation_modual(m1, n_base_filters*8, batch_normalization=batch_normalization)
    
    d2 = create_convolution_block(s1, n_base_filters*4, strides=(2, 2, 2), name="deconv", batch_normalization=batch_normalization)
    m2 = concatenate([r3, d2], axis=-1)
    s2 = creat_segmentation_modual(m2, n_base_filters*4, batch_normalization=batch_normalization)
    
    d3 = create_convolution_block(s2, n_base_filters*2, strides=(2, 2, 2), name="deconv", batch_normalization=batch_normalization)
    m3 = concatenate([r2, d3], axis=-1)
    s3 = creat_segmentation_modual(m3, n_base_filters*2, batch_normalization=batch_normalization)
    
    d4 = create_convolution_block(s3, n_base_filters, strides=(2, 2, 2), name="deconv", batch_normalization=batch_normalization)
    m4 = concatenate([r1, d4], axis=-1)
    s4 = creat_segmentation_modual(m4, n_base_filters, batch_normalization=batch_normalization)
    
    # merge multiple output
    out2 = Conv3D(n_classes, (1, 1, 1))(s2)
    out2 = UpSampling3D(size=(2, 2, 2), data_format="channels_last")(out2)
    
    out3 = Conv3D(n_classes, (1, 1, 1))(s3)
    out3 = Add()([out2, out3])
    out3 = UpSampling3D(size=(2, 2, 2), data_format="channels_last")(out3)
    
    out4 = Conv3D(n_classes, (1, 1, 1))(s4)
    out4 = Add()([out3, out4])

    outputs = Activation(activation_name)(out4)

    model = Model(inputs=inputs, outputs=outputs)
    return model


