from conv_def import *
from load_data import *
from glob import glob
import numpy as np
import time


class Unet3D(object):
    def __init__(self, sess, parameter_dict):
        # member variables
        self.input_image = None
        self.input_ground_truth = None
        self.predicted_prob = None
        self.predicted_label = None
        self.auxiliary1_prob_1x = None
        self.auxiliary2_prob_1x = None
        self.auxiliary3_prob_1x = None
        self.main_dice_loss = None
        self.auxiliary1_dice_loss = None
        self.auxiliary2_dice_loss = None
        self.auxiliary3_dice_loss = None
        self.total_dice_loss = None
        self.main_weight_loss = None
        self.auxiliary1_weight_loss = None
        self.auxiliary2_weight_loss = None
        self.auxiliary3_weight_loss = None
        self.total_weight_loss = None
        self.total_loss = None

        self.trainable_variables = None
        self.log_writer = None

        # predefined
        self.device = ['/gpu:0', '/gpu:1', '/cpu:0']
        self.sess = sess
        self.phase = parameter_dict['phase']
        self.batch_size = parameter_dict['batch_size']
        self.input_size = parameter_dict['input_size']
        self.input_channels = parameter_dict['input_channels']
        self.output_size = parameter_dict['output_size']
        self.output_channels = parameter_dict['output_channels']
        self.learning_rate = parameter_dict['learning_rate']
        self.beta1 = parameter_dict['beta1']
        self.epoch = parameter_dict['epoch']
        self.train_data_dir = parameter_dict['train_data_dir']
        self.test_data_dir = parameter_dict['test_data_dir']
        self.label_data_dir = parameter_dict['label_data_dir']
        self.model_name = parameter_dict['model_name']
        self.check_point_dir = parameter_dict['check_point_dir']
        self.resize_coefficient = parameter_dict['resize_coefficient']

        # from previous version
        self.save_interval = parameter_dict['save_interval']
        self.cube_overlapping_factor = parameter_dict['cube_overlapping_factor']

        # build model
        self.build_model()

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
            decoder1_2 = conv_bn_relu(inputs=decoder1_1, output_channels=64, kernel_size=3, stride=1,
                                      is_training=is_training, name='decoder1_2')
            feature = decoder1_2
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

        # device: cpu0
        with tf.device(device_name_or_function=self.device[2]):
            softmax_prob = tf.nn.softmax(logits=predicted_prob, name='softmax_prob')
            predicted_label = tf.argmax(input=softmax_prob, axis=4, name='argmax')

        return predicted_prob, predicted_label, auxiliary1_prob_1x, auxiliary2_prob_1x, auxiliary3_prob_1x

    '''Dice Loss, depth 3, Check'''
    @staticmethod
    def dice_loss(prediction, ground_truth):
        ground_truth = tf.one_hot(indices=ground_truth, depth=3)
        dice = 0
        for i in range(3):
            # reduce_mean calculation
            intersection = tf.reduce_mean(prediction[:, :, :, :, i] * ground_truth[:, :, :, :, i])
            union_prediction = tf.reduce_sum(prediction[:, :, :, :, i] * prediction[:, :, :, :, i])
            union_ground_truth = tf.reduce_sum(ground_truth[:, :, :, :, i] * ground_truth[:, :, :, :, i])
            union = union_ground_truth + union_prediction
            dice = dice + 2 * intersection / union
        return -dice

    '''SoftMax Loss, Check'''
    @staticmethod
    def softmax_loss(logits, labels):
        prediction = logits
        ground_truth = tf.one_hot(indices=labels, depth=3)
        softmax_prediction = tf.nn.softmax(logits=prediction)
        loss = 0
        for i in range(3):
            class_i_ground_truth = ground_truth[:, :, :, :, i]
            class_i_prediction = softmax_prediction[:, :, :, :, i]
            weight = 1 - (tf.reduce_sum(class_i_ground_truth) / tf.reduce_sum(ground_truth))
            loss = loss - tf.reduce_mean(weight * class_i_ground_truth * tf.log(
                tf.clip_by_value(t=class_i_prediction, clip_value_min=0.005, clip_value_max=1)))
        return loss

    def build_model(self):
        # input data and labels
        self.input_image = tf.placeholder(dtype=tf.float32,
                                          shape=[self.batch_size, self.input_size, self.input_size,
                                                 self.input_size, self.input_channels], name='input_image')
        self.input_ground_truth = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.input_size,
                                                                        self.input_size, self.input_size],
                                                 name='input_target')
        # probability
        self.predicted_prob, self.predicted_label, self.auxiliary1_prob_1x, \
            self.auxiliary2_prob_1x, self.auxiliary3_prob_1x = self.unet_model(self.input_image)

        # dice loss
        self.main_dice_loss = self.dice_loss(self.predicted_prob, self.input_ground_truth)
        self.auxiliary1_dice_loss = self.dice_loss(self.auxiliary1_prob_1x, self.input_ground_truth)
        self.auxiliary2_dice_loss = self.dice_loss(self.auxiliary2_prob_1x, self.input_ground_truth)
        self.auxiliary3_dice_loss = self.dice_loss(self.auxiliary3_prob_1x, self.input_ground_truth)
        self.total_dice_loss = \
            self.main_dice_loss + \
            self.auxiliary1_dice_loss * 0.8 + \
            self.auxiliary2_dice_loss * 0.4 + \
            self.auxiliary3_dice_loss * 0.2
        # class-weighted cross-entropy loss
        self.main_weight_loss = self.softmax_loss(self.predicted_prob, self.input_ground_truth)
        self.auxiliary1_weight_loss = self.softmax_loss(self.auxiliary1_prob_1x, self.input_ground_truth)
        self.auxiliary2_weight_loss = self.softmax_loss(self.auxiliary2_prob_1x, self.input_ground_truth)
        self.auxiliary3_weight_loss = self.softmax_loss(self.auxiliary3_prob_1x, self.input_ground_truth)
        self.total_weight_loss = \
            self.main_weight_loss +\
            self.auxiliary1_weight_loss * 0.9 + \
            self.auxiliary2_weight_loss * 0.6 + \
            self.auxiliary3_weight_loss * 0.3

        self.total_loss = self.total_dice_loss * 100.0 + self.total_weight_loss

        # trainable variables
        self.trainable_variables = tf.trainable_variables()

        '''No Save Module'''

    def train(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1).minimize(
            self.total_loss, var_list=self.trainable_variables
        )

        # initialization
        variables_initialization = tf.global_variables_initializer()
        self.sess.run(variables_initialization)

        # save log
        self.log_writer = tf.summary.FileWriter(logdir='../logs/', graph=self.sess.graph)

        # load all volume files
        image_list = glob(pathname='{}/*.nii.gz'.format(self.train_data_dir))
        label_list = glob(pathname='{}/*.nii.gz'.format(self.label_data_dir))
        image_data_list, label_data_list = load_image_and_label(image_list, label_list, self.resize_coefficient)

        with open(file='loss.txt', mode='w') as loss_log:
            for epoch in np.arange(self.epoch):
                start_time = time.time()

                # load batch
                train_data_batch, train_label_batch = get_image_and_label_batch(
                    image_data_list, label_data_list, self.input_size, self.batch_size)
                val_data_batch, val_label_batch = get_image_and_label_batch(
                    image_data_list, label_data_list, self.input_size, self.batch_size)

                # update network
                _, train_loss = self.sess.run(
                    [optimizer, self.total_loss],
                    feed_dict={self.input_image: train_data_batch,
                               self.input_ground_truth: train_label_batch})
                '''Summary'''
                val_loss = self.total_loss.eval({self.input_image: val_data_batch,
                                                 self.input_ground_truth: val_label_batch})
                val_prediction = self.sess.run(self.predicted_label,
                                               feed_dict={self.input_image: val_label_batch})
                # print(np.unique(train_label_batch))
                # print(np.unique(val_prediction))
                '''Dice?'''
                loss_log.write('%s %s\n' % (train_loss, val_loss))
                print(
                    'Epoch: [%2d] time: %4.4f, train_loss: %.8f, val_loss: %.8f'
                    % (epoch, time.time() - start_time, train_loss, val_loss)
                )


if __name__ == '__main__':
    sess = tf.Session()
