import os
import time
from conv_def import *
from data_io import *
from glob import glob
from json_io import *
from loss_def import *
import numpy as np

''' 3D U-Net Model '''


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
        self.fine_tuning_variables = None
        self.saver = None
        self.saver_fine_tuning = None

        # predefined
        # single-gpu
        self.device = ['/gpu:0', '/gpu:1', '/cpu:0']
        self.sess = sess
        self.parameter_dict = parameter_dict
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
        self.name_with_runtime = parameter_dict['name_with_runtime']
        self.checkpoint_dir = parameter_dict['checkpoint_dir']
        self.resize_coefficient = parameter_dict['resize_coefficient']

        # from previous version
        self.save_interval = parameter_dict['save_interval']
        self.cube_overlapping_factor = parameter_dict['cube_overlapping_factor']

        # build model
        self.build_model()

    def unet_model(self, inputs):
        is_training = (self.phase == 'train')
        concat_dimension = 4  # channels_last

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
        # TODO: draw a graph

        # device: cpu0
        with tf.device(device_name_or_function=self.device[2]):
            softmax_prob = tf.nn.softmax(logits=predicted_prob, name='softmax_prob')
            predicted_label = tf.argmax(input=softmax_prob, axis=4, name='predicted_label')

        return predicted_prob, predicted_label, auxiliary1_prob_1x, auxiliary2_prob_1x, auxiliary3_prob_1x

    def build_model(self):
        # input data and labels
        self.input_image = tf.placeholder(dtype=tf.float32,
                                          shape=[self.batch_size, self.input_size, self.input_size,
                                                 self.input_size, self.input_channels], name='input_image')
        self.input_ground_truth = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.input_size,
                                                                        self.input_size, self.input_size],
                                                 name='input_ground_truth')
        # probability
        self.predicted_prob, self.predicted_label, self.auxiliary1_prob_1x, \
            self.auxiliary2_prob_1x, self.auxiliary3_prob_1x = self.unet_model(self.input_image)

        # dice loss
        self.main_dice_loss = dice_loss_function(self.predicted_prob, self.input_ground_truth)
        self.auxiliary1_dice_loss = dice_loss_function(self.auxiliary1_prob_1x, self.input_ground_truth)
        self.auxiliary2_dice_loss = dice_loss_function(self.auxiliary2_prob_1x, self.input_ground_truth)
        self.auxiliary3_dice_loss = dice_loss_function(self.auxiliary3_prob_1x, self.input_ground_truth)
        self.total_dice_loss = \
            self.main_dice_loss + \
            self.auxiliary1_dice_loss * 0.8 + \
            self.auxiliary2_dice_loss * 0.4 + \
            self.auxiliary3_dice_loss * 0.2
        # class-weighted cross-entropy loss
        self.main_weight_loss = softmax_loss_function(self.predicted_prob, self.input_ground_truth)
        self.auxiliary1_weight_loss = softmax_loss_function(self.auxiliary1_prob_1x, self.input_ground_truth)
        self.auxiliary2_weight_loss = softmax_loss_function(self.auxiliary2_prob_1x, self.input_ground_truth)
        self.auxiliary3_weight_loss = softmax_loss_function(self.auxiliary3_prob_1x, self.input_ground_truth)
        self.total_weight_loss = \
            self.main_weight_loss +\
            self.auxiliary1_weight_loss * 0.9 + \
            self.auxiliary2_weight_loss * 0.6 + \
            self.auxiliary3_weight_loss * 0.3

        # TODO: adjust the weights
        self.total_loss = self.total_dice_loss * 1e5 + self.total_weight_loss

        # trainable variables
        self.trainable_variables = tf.trainable_variables()

        # TODO: how to extract layers for fine-tuning? why?
        '''How to list all of them'''
        fine_tuning_layer = [
                'encoder1_1/encoder1_1_conv/kernel:0',
                'encoder1_2/encoder1_2_conv/kernel:0',
                'encoder2_1/encoder2_1_conv/kernel:0',
                'encoder2_2/encoder2_2_conv/kernel:0',
                'encoder3_1/encoder3_1_conv/kernel:0',
                'encoder3_2/encoder3_2_conv/kernel:0',
                'encoder4_1/encoder4_1_conv/kernel:0',
                'encoder4_2/encoder4_2_conv/kernel:0',
        ]

        # TODO: what does this part mean
        self.fine_tuning_variables = []
        for variable in self.trainable_variables:
            # print('\'%s\',' % variable.name)
            for index, kernel_name in enumerate(fine_tuning_layer):
                if kernel_name in variable.name:
                    self.fine_tuning_variables.append(variable)
                    break  # not necessary to continue

        self.saver = tf.train.Saver()
        self.saver_fine_tuning = tf.train.Saver(self.fine_tuning_variables)
        # The Saver class adds ops to save and restore variables to and from checkpoints.
        # It also provides convenience methods to run these ops.
        print('Model built successfully.')

    def save_checkpoint(self, checkpoint_dir, model_name, global_step):
        model_dir = '%s_%s' % (self.batch_size, self.output_size)
        '''Why?'''
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=global_step)
        # defaults to the list of all saveable objects

    '''To be checked!'''

    def load_checkpoint(self, checkpoint_dir):
        model_dir = '%s_%s' % (self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        checkpoint_state = tf.train.get_checkpoint_state(checkpoint_dir)
        # A CheckpointState if the state was available, None otherwise.
        if checkpoint_state and checkpoint_state.model_checkpoint_path:
            checkpoint_name = os.path.basename(checkpoint_state.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, checkpoint_name))
            return True
        else:
            return False

    '''A function for fine-tuning'''

    def train(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1).minimize(
            self.total_loss, var_list=self.trainable_variables
        )

        # initialization
        variables_initialization = tf.global_variables_initializer()
        self.sess.run(variables_initialization)

        # TODO: load pre-trained model
        # TODO: load checkpoint

        # save log
        if not os.path.exists('logs/'):
            os.makedirs('logs/')
        self.log_writer = tf.summary.FileWriter(logdir='logs/', graph=self.sess.graph)

        # load all volume files
        image_list = glob(pathname='{}/*.nii.gz'.format(self.train_data_dir))
        label_list = glob(pathname='{}/*.nii.gz'.format(self.label_data_dir))
        image_data_list, label_data_list = load_image_and_label(image_list, label_list, self.resize_coefficient)
        print('Data loaded successfully.')

        if not os.path.exists('loss/'):
            os.makedirs('loss/')
        line_buffer = 1
        with open(file='loss/loss_'+self.name_with_runtime+'.txt', mode='w', buffering=line_buffer) as loss_log:
            loss_log.write(dict_to_json(self.parameter_dict))

            for epoch in np.arange(self.epoch):
                start_time = time.time()

                # load batch
                train_data_batch, train_label_batch = get_image_and_label_batch(
                    image_data_list, label_data_list, self.input_size, self.batch_size)
                val_data_batch, val_label_batch = get_image_and_label_batch(
                    image_data_list, label_data_list, self.input_size, self.batch_size)
                '''The same data at this stage'''

                # update network
                _, train_loss, dice_loss, weight_loss= self.sess.run(
                    [optimizer, self.total_loss, self.total_dice_loss, self.total_weight_loss],
                    feed_dict={self.input_image: train_data_batch,
                               self.input_ground_truth: train_label_batch})
                '''Summary'''
                # may not run each time
                val_loss = self.total_loss.eval({self.input_image: val_data_batch,
                                                 self.input_ground_truth: val_label_batch})
                val_prediction = self.sess.run(self.predicted_label,
                                               feed_dict={self.input_image: val_data_batch})

                loss_log.write('[label] ')
                loss_log.write(str(np.unique(train_label_batch)))
                loss_log.write(str(np.unique(val_label_batch)))
                loss_log.write(str(np.unique(val_prediction)))
                loss_log.write('\n')

                # Dice
                dice = []
                for i in range(self.output_channels):
                    intersection = np.sum(
                        ((val_label_batch[:, :, :, :] == i) * 1) * ((val_prediction[:, :, :, :] == i) * 1)
                    )
                    union = np.sum(
                        ((val_label_batch[:, :, :, :] == i) * 1) + ((val_prediction[:, :, :, :] == i) * 1)
                    ) + 1e-5
                    '''Why not necessary to square'''
                    dice.append(2.0 * intersection / union)
                loss_log.write('[Dice] %s \n' % dice)

                # loss_log.write('%s %s\n' % (train_loss, val_loss))
                output_format = '[Epoch] %d, time: %4.4f, train_loss: %.8f, val_loss: %.8f \n' \
                                '[Loss] dice_loss: %.8f, weight_loss: %.8f \n\n'\
                                % (epoch, time.time() - start_time, train_loss, val_loss,
                                   dice_loss * 1e5, weight_loss)
                loss_log.write(output_format)
                print(output_format, end='')
                if np.mod(epoch+1, self.save_interval) == 0:
                    self.save_checkpoint(self.checkpoint_dir, self.model_name, global_step=epoch+1)
                    print('Model saved with epoch %d' % (epoch+1))


if __name__ == '__main__':
    sess = tf.Session()
