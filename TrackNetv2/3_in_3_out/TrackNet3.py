from keras.models import *
from keras.layers import *
from keras.activations import *

def TrackNet3( input_height, input_width ): #input_height = 288, input_width = 512

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

	#Layer12
	x = ( Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer13
	x = ( Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer14
	x = concatenate( [UpSampling2D( (2,2), data_format='channels_first')(x), x3], axis=1)

	#Layer15
	x = ( Conv2D( 256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer16
	x = ( Conv2D( 256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer17
	x = ( Conv2D( 256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)
	
	#Layer18
	x = concatenate( [UpSampling2D( (2,2), data_format='channels_first')(x), x2], axis=1)

	#Layer19
	x = ( Conv2D( 128 , (3, 3), kernel_initializer='random_uniform', padding='same' , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer20
	x = ( Conv2D( 128 , (3, 3), kernel_initializer='random_uniform', padding='same' , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer21
	x = concatenate( [UpSampling2D( (2,2), data_format='channels_first')(x), x1], axis=1)

	#Layer22
	x = ( Conv2D( 64 , (3, 3), kernel_initializer='random_uniform', padding='same'  , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer23
	x = ( Conv2D( 64 , (3, 3), kernel_initializer='random_uniform', padding='same'  , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer24
	x =  Conv2D( 3 , (1, 1) , kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('sigmoid'))(x)
        

	o_shape = Model(imgs_input , x ).output_shape

	#print ("layer24 output shape:", o_shape[1],o_shape[2],o_shape[3])
	#Layer24 output shape: (3, 288, 512)

	OutputHeight = o_shape[2]
	OutputWidth = o_shape[3]

	output = x

	model = Model( imgs_input , output)
	#model input unit:9*288*512, output unit:3*288*512
	model.outputWidth = OutputWidth
	model.outputHeight = OutputHeight

	#Show model's details
	#model.summary()

	return model




