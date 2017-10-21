import datetime
import os
import tensorflow as tf
from json_io import *
from model import Unet3D

''' Main Function '''


def init_parameter(name):
    # dictionary
    parameter_dict = dict()
    # current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    parameter_dict['phase'] = "train"
    parameter_dict['batch_size'] = 1
    parameter_dict['input_size'] = 96
    parameter_dict['input_channels'] = 1
    parameter_dict['output_size'] = 96
    parameter_dict['output_channels'] = 3
    parameter_dict['learning_rate'] = 0.001
    parameter_dict['beta1'] = 0.5
    parameter_dict['epoch'] = 100000
    parameter_dict['train_data_dir'] = "../hvsmr/data/"
    parameter_dict['test_data_dir'] = "../hvsmr/data/"
    parameter_dict['label_data_dir'] = "../hvsmr/label"
    parameter_dict['model_name'] = "hvsmr_" + name + ".model"
    parameter_dict['name_with_runtime'] = name
    parameter_dict['checkpoint_dir'] = "checkpoint/"
    parameter_dict['resize_coefficient'] = 1.0
    # from previous version
    parameter_dict['save_interval'] = 2500
    parameter_dict['cube_overlapping_factor'] = 4

    return parameter_dict


def main(args):
    # set cuda visible device
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

    # name the model
    name = 'test'
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    name = name + '_' + current_time

    # load predefined training data
    parameter_dict = init_parameter(name)
    if not os.path.exists('json/'):
        os.makedirs('json/')
    parameter_json = dict_to_json(parameter_dict, write_file=True, file_name='json/parameter_' + name + '.json')
    print(parameter_json)

    # gpu processing, for further set
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95, allow_growth=True)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
        model = Unet3D(sess=sess, parameter_dict=parameter_dict)
        if parameter_dict['phase'] == 'train':
            model.train()


if __name__ == '__main__':
    tf.app.run()
