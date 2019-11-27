from keras.layers import *
from keras.models import Model


def DeconvNet(shape, denoising=False):
    input_tensor = Input(shape)
    if False:
        x = GaussianNoise(0.5)(input_tensor)
        x = Conv2D(64, (3, 3),
                   activation='relu',
                   padding='same',
                   name='block1_conv1')(x)
    else:
        x = Conv2D(64, (3, 3),
                   activation='relu',
                   padding='same',
                   name='block1_conv1')(input_tensor)

    x = Conv2D(64, (3, 3),
               activation='relu',
               padding='same',
               name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3),
               activation='relu',
               padding='same',
               name='block2_conv1')(x)
    x = Conv2D(128, (3, 3),
               activation='relu',
               padding='same',
               name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv1')(x)
    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv2')(x)
    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv1')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv1')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    block_shape = K.int_shape(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(block_shape[1] * block_shape[2] * 84, activation='relu')(x)
    x = Reshape((block_shape[1], block_shape[2], 84))(x)

    x = UpSampling2D()(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_upconv1')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_upconv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_upconv3')(x)
    x = UpSampling2D()(x)
    # Block 2
    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_upconv1')(x)
    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_upconv2')(x)
    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_upconv3')(x)
    x = UpSampling2D()(x)

    x = Conv2D(128, (3, 3),
               activation='relu',
               padding='same',
               name='block2_upconv1')(x)
    x = Conv2D(128, (3, 3),
               activation='relu',
               padding='same',
               name='block2_upconv2')(x)
    x = UpSampling2D()(x)
    x = Conv2D(64, (3, 3),
               activation='relu',
               padding='same',
               name='block1_upconv1')(x)
    x = Conv2D(64, (3, 3),
               activation='relu',
               padding='same',
               name='block1_upconv2')(x)
    x = UpSampling2D()(x)
    x = Conv2D(32, (3, 3),
               activation='relu',
               padding='same',
               name='pre_block')(x)
    output = Conv2D(3, (3, 3),
                    activation='relu',
                    padding='same',
                    name='output')(x)
    model = Model(input_tensor, output)
    model.summary()
    return model


def UNet(shape, denoising=False):
    inputs = Input(shape)
    if False:
        x = GaussianNoise(0.5)(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(x)
    else:
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)

    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)

    up6 = Conv2D(
        512,
        2,
        activation='relu',
        padding='same')(
        UpSampling2D(
            size=(
                2,
                2))(conv5))
    merge6 = Concatenate()([conv4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2D(
        256,
        2,
        activation='relu',
        padding='same')(
        UpSampling2D(
            size=(
                2,
                2))(conv6))
    merge7 = Concatenate()([conv3, up7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = Conv2D(
        128,
        2,
        activation='relu',
        padding='same')(
        UpSampling2D(
            size=(
                2,
                2))(conv7))
    merge8 = Concatenate()([conv2, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = Conv2D(
        64,
        2,
        activation='relu',
        padding='same')(
        UpSampling2D(
            size=(
                2,
                2))(conv8))
    merge9 = Concatenate()([conv1, up9])
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same')(conv9)
    conv10 = Conv2D(3, 1, activation='relu')(conv9)

    model = Model(input=inputs, output=conv10)
    model.summary()
    return model


def MSUnet(shape):
    pass
