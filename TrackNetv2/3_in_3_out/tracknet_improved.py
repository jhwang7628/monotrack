from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.losses import *
import tensorflow.keras.backend as K
import tensorflow as tf
import tensorflow.keras as keras

from focal_loss import binary_focal_loss

def focal_tversky_loss(y_true, y_pred, alpha=0.5, beta=0.5, gamma=2, smooth=1e-6):
    #True Positives, False Positives & False Negatives
    TP = K.sum((y_pred * y_true))
    FP = K.sum(((1-y_true) * y_pred))
    FN = K.sum((y_true * (1-y_pred)))

    tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
    focal_tversky = K.pow((1 - tversky), gamma)

    return focal_tversky
    
def dice_loss(y_true, y_pred, smooth=1e-4):
    intersection = K.sum(y_true * y_pred, axis=-1) + smooth
    union = K.sum(y_true + y_pred, axis=-1) + 2. * smooth
    return 1 - 2. * intersection / union

def regressor_loss(y_true, y_pred):
#     mse = K.sum((1 - y_true[:, :, 2]) * (1 - y_pred[:, :, 2]) * K.sum(K.square(y_true[:, :, :2] - y_pred[:, :, :2]), axis=-1), axis=-1)
#     mae = K.sum(K.abs(y_true[:, :, 2] - y_pred[:, :, 2]), axis=-1)
#     return K.mean(mse + mae)
    return mean_squared_error(y_true, y_pred)

def regressor_metric(y_true, y_pred):
#     mse = K.sum((1 - y_true[:, :, 2]) * K.sum(K.square(y_true[:, :, :2] - y_pred[:, :, :2]), axis=-1), axis=-1)
#     return K.mean(mse)
    return mean_squared_error(y_true, y_pred)

#Loss function
# def custom_loss(y_true, y_pred): #hm_true, hm_pred
#     #pos_mask = tf.cast(tf.equal(hm_true, 1), tf.float32)
#     #neg_mask = tf.cast(tf.less(hm_true, 1), tf.float32)
#     #neg_weights = tf.pow(1 - hm_true, 4)
#     #pos_loss = -tf.log(tf.clip_by_value(hm_pred, 1e-4, 1. - 1e-4)) * tf.pow(1 - hm_pred, 2) * pos_mask
#     #neg_loss = -tf.log(tf.clip_by_value(1 - hm_pred, 1e-4, 1. - 1e-4)) * tf.pow(hm_pred, 2) * neg_weights * neg_mask
#     #num_pos = tf.reduce_sum(pos_mask)
#     #pos_loss = tf.reduce_sum(pos_loss)
#     #neg_loss = tf.reduce_sum(neg_loss)
#     #loss = tf.cond(tf.greater(num_pos, 0), lambda: (pos_loss + neg_loss) / num_pos, lambda: neg_loss)
#     loss = (-1)*(K.square(1 - y_pred) * y_true * K.log(K.clip(y_pred, K.epsilon(), 1)) + K.square(y_pred) * (1 - y_true) * K.log(K.clip(1 - y_pred, K.epsilon(), 1)))
#     #gamma=2
#     #alpha=0.25
#     #y_true = tf.cast(y_true, tf.float32)
#     #alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*(1-alpha)
#     #p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred) + K.epsilon()
#     #focal_loss = - alpha_t * K.pow((K.ones_like(y_true)-p_t),gamma) * K.log(p_t)
#     #return K.mean(focal_loss)
#     return (loss)

# Custom loss function
def custom_loss(y_true, y_pred, gamma=2):
    y_pred = K.flatten(y_pred)
    y_true = K.flatten(y_true)
#     return dice_loss(y_true, y_pred)
    return binary_focal_loss(y_true, y_pred, gamma)
#     return focal_tversky_loss(y_true, y_pred, alpha=0.1, beta=0.9, gamma=gamma)
#     return 420*binary_focal_loss(y_true, y_pred, gamma) + dice_loss(y_true, y_pred)
    # return 10 * binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def conv_block(x, filters, size=3, channel_first=True):
    x = Conv2D(
        filters, (size, size), 
        kernel_initializer='he_uniform', padding='same', 
        data_format='channels_first' if channel_first else 'channels_last')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def down_block(x, filters, repeats, use_res=False, channel_first=True):
    for i in range(repeats):
        x = conv_block(x, filters, 3, channel_first)
        if use_res and i == 0:
            x0 = x
    if use_res:
        y = MaxPooling2D(
            (2, 2), strides=(2, 2), 
            data_format='channels_first' if channel_first else 'channels_last')(x + x0)
    else:
        y = MaxPooling2D(
            (2, 2), strides=(2, 2), 
            data_format='channels_first' if channel_first else 'channels_last')(x)
    return y, x
    
def up_block(x, xu, filters, repeats, use_res=False, channel_first=True):
    x = concatenate([
            Conv2DTranspose(
                filters, 
                kernel_size=(2, 2), strides=(2, 2), 
                data_format='channels_first' if channel_first else 'channels_last')(x), 
            xu], axis=1)
    for i in range(repeats):
        x = conv_block(x, filters)
        if use_res and i == 0:
            x0 = x
    if use_res:
        x = x + x0
    return x

def double_conv_layer(x, filter_size, size, dropout = 0, batch_norm=False):
    conv = Conv2D(size, (filter_size, filter_size), padding='same')(x)
    if batch_norm is True:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(size, (filter_size, filter_size), padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    if dropout > 0:
        conv = Dropout(dropout)(conv)

    shortcut = Conv2D(size, kernel_size=(1, 1), padding='same')(x)
    if batch_norm is True:
        shortcut = BatchNormalization()(shortcut)

    res_path = add([shortcut, conv])
    return res_path

def gating_signal(inp, out_size, batch_norm=False):
    x = Conv2D(out_size, (1, 1), padding='same')(inp)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def expend_as(tensor, rep):
     return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                          arguments={'repnum': rep})(tensor)
    
def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

    theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

    phi_g = Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same')(phi_g)  # 16

    concat_xg = add([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    upsample_psi = expend_as(upsample_psi, shape_x[3])

    y = multiply([upsample_psi, x])

    result = Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = BatchNormalization()(result)
    return result_bn

class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                teacher_predictions / self.temperature,
                student_predictions / self.temperature
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results
    
def TrackNetOriginal(input_height, input_width, num_consec=3, grayscale=True):
    if grayscale:
        imgs_input = Input(shape=(num_consec, input_height, input_width))
        x = imgs_input
    else:
        imgs_input = Input(shape=(num_consec, input_height, input_width, 3))
        x = K.permute_dimensions(imgs_input, (0, 1, 4, 2, 3))
        x = Reshape(target_shape=(num_consec * 3, input_height, input_width))(x)
    
    # Down block
    x, x1 = down_block(x, 64, 2)
    x, x2 = down_block(x, 128, 2)
    x, x3 = down_block(x, 256, 3)
    
    # Bottom block
    for i in range(3):
        x = conv_block(x, 512)
    
    # Up block
    x = up_block(x, x3, 256, 3)
    x = up_block(x, x2, 128, 2)
    x = up_block(x, x1, 64, 2)
    
    # Final output
    x = Conv2D( num_consec, (1, 1) , kernel_initializer='he_uniform', padding='same', data_format='channels_first' )(x)
    x = Activation('sigmoid', dtype='float32')(x)
        
    output = x
    model = Model(imgs_input , output)
    model.summary()

    return model

def TrackNetImproved(input_height, input_width, num_consec=3, grayscale=True):
    if grayscale:
        imgs_input = Input(shape=(num_consec, input_height, input_width))
        x = imgs_input
    else:
        imgs_input = Input(shape=(num_consec, input_height, input_width, 3))
        x = K.permute_dimensions(imgs_input, (0, 1, 4, 2, 3))
        x = Reshape(target_shape=(num_consec * 3, input_height, input_width))(x)
    
    # Down block
    x, x1 = down_block(x, 32, 3, use_res=True)
    x, x2 = down_block(x, 64, 3, use_res=True)
    x, x3 = down_block(x, 128, 3, use_res=True)
    
    # Bottom block
    for i in range(3):
        x = conv_block(x, 256)
    
    # Up block
    x = up_block(x, x3, 128, 3, use_res=True)
    x = up_block(x, x2, 64, 3, use_res=True)
    x = up_block(x, x1, 32, 3)
    
    # Final output
    x = Conv2D(num_consec, (1, 1), 
               kernel_initializer='he_uniform', 
               padding='same', 
               data_format='channels_first',
               bias_initializer=tf.keras.initializers.constant(-3.2))(x)
    
    x = Activation('sigmoid', dtype='float32')(x)
        
    output = x
    model = Model(imgs_input , output)
    model.summary()

    return model

def AttentionTrackNet(input_height, input_width, num_consec=3, grayscale=True, dropout_rate=0.0, batch_norm=True):
    if grayscale:
        imgs_input = Input(shape=(num_consec, input_height, input_width))
        x = K.permute_dimensions(imgs_input, (0, 2, 3, 1))
    else:
        imgs_input = Input(shape=(num_consec, input_height, input_width, 3))
        x = K.permute_dimensions(imgs_input, (0, 2, 3, 4, 1))
        x = Reshape(target_shape=(input_height, input_width, num_consec * 3))(x)

    # Downsampling layers
    # DownRes 1, double residual convolution + pooling
    FILTER_SIZE = 3
    FILTER_NUM = 32
    UP_SAMP_SIZE = 2
    OUTPUT_MASK_CHANNEL = num_consec
    
    conv_64 = double_conv_layer(x, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = double_conv_layer(pool_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = double_conv_layer(pool_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = double_conv_layer(pool_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 4*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 4*FILTER_NUM)
    up_16 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = concatenate([up_16, att_16], axis=-1)
    up_conv_16 = double_conv_layer(up_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 2*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 2*FILTER_NUM)
    up_32 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = concatenate([up_32, att_32], axis=-1)
    up_conv_32 = double_conv_layer(up_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    gating_64 = gating_signal(up_conv_32, FILTER_NUM, batch_norm)
    att_64 = attention_block(conv_64, gating_64, FILTER_NUM)
    up_64 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = concatenate([up_64, att_64], axis=-1)
    up_conv_64 = double_conv_layer(up_64, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # 1*1 convolutional layers
    # sigmoid nonlinear activation
    conv_final = Conv2D(
        OUTPUT_MASK_CHANNEL, kernel_size=(1,1), 
        bias_initializer=tf.keras.initializers.constant(-3.2))(up_conv_64)
    conv_final = Activation('sigmoid')(conv_final)
    conv_final = K.permute_dimensions(conv_final, (0, 3, 1, 2))

    # Model integration
    model = Model(imgs_input, conv_final, name="AttentionTrackNet")
    model.summary()
    return model

def TrackNetRegressor(input_height, input_width, num_consec=3, grayscale=True):
    if grayscale:
        imgs_input = Input(shape=(num_consec, input_height, input_width))
        x = imgs_input
    else:
        imgs_input = Input(shape=(num_consec, input_height, input_width, 3))
        x = K.permute_dimensions(imgs_input, (0, 1, 4, 2, 3))
        x = Reshape(target_shape=(num_consec * 3, input_height, input_width))(x)
    
    kwargs     = {'activation': 'relu', 'padding': 'same', 'data_format': 'channels_first'}
    conv_drop  = 0.2
    dense_drop = 0.5

    with_dropout = False
    
    x = Conv2D(256, (9, 9), **kwargs)(x)
#     x0 = Conv2D(64, (3, 3), **kwargs)(x)
#     x = BatchNormalization()(x0)
#     if with_dropout: x = Dropout(conv_drop)(x)

#     x1 = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first')(x+x0)
#     x = Conv2D(64, (3, 3), **kwargs)(x1)
#     x = Conv2D(64, (3, 3), **kwargs)(x)
#     x = BatchNormalization()(x)
#     if with_dropout: x = Dropout(conv_drop)(x)

#     x2 = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first')(x+x1)
#     x = Conv2D(64, (3, 3), **kwargs)(x2)
#     x = Conv2D(64, (3, 3), **kwargs)(x)
# #     x = BatchNormalization()(x)
#     if with_dropout: x = Dropout(conv_drop)(x)
        
#     x3 = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first')(x+x2)
#     x = Conv2D(64, (3, 3), **kwargs)(x3)
#     x = Conv2D(64, (3, 3), **kwargs)(x)
# #     x = BatchNormalization()(x)
#     if with_dropout: x = Dropout(conv_drop)(x)

#     x4 = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first')(x+x3)
#     x = Conv2D(64, (3, 3), **kwargs)(x4)
#     x = Conv2D(64, (3, 3), **kwargs)(x)
# #     x = BatchNormalization()(x)
#     if with_dropout: x = Dropout(conv_drop)(x)

#     x5 = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first')(x+x4)
#     x = Conv2D(64, (3, 3), **kwargs)(x5)
#     x = Conv2D(64, (3, 3), **kwargs)(x)
# #     x = BatchNormalization()(x)
#     if with_dropout: x = Dropout(conv_drop)(x)

    h = MaxPooling2D(pool_size=(1, int(x.shape[3])), data_format='channels_first')(x)
    if with_dropout: h = Dropout(dense_drop)(h)
    h = Dense(4, activation='relu')(h)
    h = Flatten()(h)

    v = MaxPooling2D(pool_size=(int(x.shape[2]), 1), data_format='channels_first')(x)
    v = Flatten()(v)
    if with_dropout: v = Dropout(dense_drop)(v)
    v = Dense(4, activation='relu')(v)
    h = Flatten()(h)

    x = Concatenate()([h,v])
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    if with_dropout: x = Dropout(0.5)(x)
        
    x = Dense(3 * num_consec, activation='sigmoid')(x)
    x = Reshape(target_shape=(num_consec, 3))(x)
    
    model = Model(imgs_input, x)
    model.summary()
    return model

def TrackNetRegressorII(num_consec=3):
    tracknet_model = load_model('./model906_30', custom_objects={'custom_loss': custom_loss})
    tracknet_model.trainable = False
    tracknet_model.summary()
    for layer in tracknet_model.layers:
        layer.trainable = False
        
    x = tracknet_model.layers[33].output
    x = Conv2D(32, (1, 1) , kernel_initializer='he_uniform', padding='same', data_format='channels_first')(x)
    x = Flatten()(x)
    
    coords = Dense(16, activation='relu')(x)
    coords = Dense(16, activation='relu')(coords)
    coords = Dense(16, activation='relu')(coords)
    coords = Dense(2 * num_consec, activation='sigmoid')(coords)
    coords = Reshape(target_shape=(num_consec, 2), name='coords')(coords)
    
    exists = Dense(16, activation='relu')(x)
    exists = Dense(16, activation='relu')(exists)
    exists = Dense(16, activation='relu')(exists)
    exists = Dense(num_consec, activation='sigmoid', dtype='float32', name='exists')(exists)
    
    model = Model(tracknet_model.input, [coords, exists])
    model.summary()
    return model