from keras.models import *
from keras.layers import *
from keras.activations import *

def HitNet( input_height, input_width ): #input_height = 288, input_width = 512

    imgs_input = Input(shape=(9,input_height,input_width))

    #Layer1
    x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(imgs_input)
    x = ( Activation('relu'))(x)
    x = ( BatchNormalization())(x)

    #Layer2
    x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
    x = ( Activation('relu'))(x)
    x1 = ( BatchNormalization())(x)

    #Layer3
    x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x1)

    #Layer4
    x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
    x = ( Activation('relu'))(x)
    x = ( BatchNormalization())(x)

    #Layer5
    x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
    x = ( Activation('relu'))(x)
    x2 = ( BatchNormalization())(x)

    #Layer6
    x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x2)

    #Layer7
    x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
    x = ( Activation('relu'))(x)
    x = ( BatchNormalization())(x)

    #Layer8
    x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
    x = ( Activation('relu'))(x)
    x = ( BatchNormalization())(x)

    #Layer9
    x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
    x = ( Activation('relu'))(x)
    x3 = ( BatchNormalization())(x)

    #Layer10
    x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x3)

    #Layer11
    x = ( Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
    x = ( Activation('relu'))(x)
    x = ( BatchNormalization())(x)

    #Last layers (train layers from this point)
    #Layer12
    x = ( Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
    x = ( Activation('relu'))(x)
    x = ( BatchNormalization())(x)

    #Layer13
    x = ( Conv2D(16, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
    x = ( Activation('relu'))(x)
    x = ( BatchNormalization())(x)

    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)

    o_shape = Model(imgs_input, x ).output_shape

    print ("Final layer output shape:", o_shape)
    
    output = x

    model = Model(imgs_input , output)
    model.ncustom = 8

    return model