from conv_def import *


class Unet3D(object):
    def __init__(self, sess, parameter):
        self.sess = sess
        self.phase = parameter['phase']
        self.output_channels = parameter['output_channels']
        self.device = ['/gpu:0', '/gpu:1', '/cpu:0']

    def unet_model(self, inputs):
        is_training = (self.phase == 'train')
        concat_dimension = 4  # channels_last
        '''What does concat dimension mean'''
        # down-sampling path
        # device: gpu0
        with tf.device(device_name_or_function=self.device[0]):
            # first level
            encoder1_1 = conv_bn_relu(inputs=inputs, output_channels=32, kernel_size=3, stride=1,
                                   is_training=is_training, name='encoder1_1')
            encoder1_2 = conv_bn_relu(inputs=encoder1_1, output_channels=64, kernel_size=3, stride=1,
                                   is_training=is_training, name='encoder1_2')
            pool1 = tf.layers.max_pooling3d(
                inputs=encoder1_2,
                pool_size=2,                    # pool_depth, pool_height, pool_width
                strides=2,
                padding='valid',                # No padding, default
                data_format='channels_last',    # default
                name='pool1'
            )
            # second level
            encoder2_1 = conv_bn_relu(inputs=pool1, output_channels=64, kernel_size=3, stride=1,
                                   is_training=is_training, name='encoder2_1')
            encoder2_2 = conv_bn_relu(inputs=encoder2_1, output_channels=128, kernel_size=3, stride=1,
                                   is_training=is_training, name='encoder2_2')
            pool2 = tf.layers.max_pooling3d(inputs=encoder2_2, pool_size=2, strides=2, name='pool2')
            # third level
            encoder3_1 = conv_bn_relu(inputs=pool2, output_channels=128, kernel_size=3, stride=1,
                                   is_training=is_training, name='encoder3_1')
            encoder3_2 = conv_bn_relu(inputs=encoder3_1, output_channels=256, kernel_size=3, stride=1,
                                   is_training=is_training, name='encoder3_2')
            pool3 = tf.layers.max_pooling3d(inputs=encoder3_2, pool_size=2, strides=2, name='pool3')
            # forth level
            encoder4_1 = conv_bn_relu(inputs=pool3, output_channels=256, kernel_size=3, stride=1,
                                   is_training=is_training, name='encoder4_1')
            encoder4_2 = conv_bn_relu(inputs=encoder4_1, output_channels=512, kernel_size=3, stride=1,
                                   is_training=is_training, name='encoder4_2')
            bottom = encoder4_2

        # up-sampling path
        # device: gpu1
        with tf.device(device_name_or_function=self.device[1]):
            # third level
            deconv3 = deconv_bn_relu(inputs=bottom, output_channels=512, is_training=is_training,
                                       name='deconv3')
            concat_3 = tf.concat([deconv3, encoder3_2], axis=concat_dimension, name='concat_3')
            decoder3_1 = conv_bn_relu(inputs=concat_3, output_channels=256, kernel_size=3, stride=1,
                                       is_training=is_training, name='decoder3_1')
            decoder3_2 = conv_bn_relu(inputs=decoder3_1, output_channels=256, kernel_size=3, stride=1,
                                  is_training=is_training, name='decoder3_2')
            # second level
            deconv2 = deconv_bn_relu(inputs=decoder3_2, output_channels=256, is_training=is_training,
                                     name='deconv2')
            concat_2 = tf.concat([deconv2, encoder2_2], axis=concat_dimension, name='concat_2')
            decoder2_1 = conv_bn_relu(inputs=concat_2, output_channels=128, kernel_size=3, stride=1,
                                  is_training=is_training, name='decoder2_1')
            decoder2_2 = conv_bn_relu(inputs=decoder2_1, output_channels=128, kernel_size=3, stride=1,
                                  is_training=is_training, name='decoder2_2')
            # first level
            deconv1 = deconv_bn_relu(inputs=decoder2_2, output_channels=128, is_training=is_training,
                                     name='deconv1')
            concat_1 = tf.concat([deconv1, encoder1_2], axis=concat_dimension, name='concat_1')
            decoder1_1 = conv_bn_relu(inputs=concat_1, output_channels=64, kernel_size=3, stride=1,
                                      is_training=is_training, name='decoder1_1')
            decoder2_2 = conv_bn_relu(inputs=decoder1_1, output_channels=64, kernel_size=3, stride=1,
                                      is_training=is_training, name='decoder1_2')
            feature = decoder2_2
            # predicted probability
            predicted_prob = conv3d(inputs=feature, output_channels=self.output_channels, kernel_size=1,
                                    stride=1, use_bias=True, name='predicted_prob')

            '''auxiliary prediction'''
            # forth level
            auxiliary3_prob_8x = conv3d(inputs=encoder4_2, output_channels=self.output_channels, kernel_size=1,
                                stride=1, use_bias=True, name='auxiliary3_prob_8x')
            auxiliary3_prob_4x = deconv3d(inputs=auxiliary3_prob_8x, output_channels=self.output_channels,
                                          name='auxiliary3_prob_4x')
            auxiliary3_prob_2x = deconv3d(inputs=auxiliary3_prob_4x, output_channels=self.output_channels,
                                        name='auxiliary3_prob_2x')
            auxiliary3_prob_1x = deconv3d(inputs=auxiliary3_prob_2x, output_channels=self.output_channels,
                                        name='auxiliary3_prob_1x')
            # third level
            auxiliary2_prob_4x = conv3d(inputs=decoder3_2, output_channels=self.output_channels, kernel_size=1,
                                        stride=1, use_bias=True, name='auxiliary2_prob_4x')
            auxiliary2_prob_2x = deconv3d(inputs=auxiliary2_prob_4x, output_channels=self.output_channels,
                                        name='auxiliary2_prob_2x')
            auxiliary2_prob_1x = deconv3d(inputs=auxiliary2_prob_2x, output_channels=self.output_channels,
                                      name='auxiliary2_prob_1x')
            # second level
            auxiliary1_prob_2x = conv3d(inputs=decoder2_2, output_channels=self.output_channels, kernel_size=1,
                                        stride=1, use_bias=True, name='auxiliary1_prob_2x')
            auxiliary1_prob_1x = deconv3d(inputs=auxiliary1_prob_2x, output_channels=self.output_channels,
                                          name='auxiliary1_prob_1x')

        with tf.device(device_name_or_function=self.device[2]):
            softmax_prob = tf.nn.softmax(logits=predicted_prob, name='softmax_prob')
            predicted_label = tf.argmax(input=softmax_prob, axis=4, name='argmax')

        return predicted_prob, predicted_label, auxiliary1_prob_1x, auxiliary2_prob_1x, auxiliary3_prob_1x
