import numpy as np

import pixelizer
import helper
from normalization import normalize_output, normalize_input
from network import Network


def convert_all_images(number):
    pixelizer.convert_images(helper.get_all_filenames('Dataset/Training_set/Default_set/'), number)


def raw_data_to_points(data):
    points_arr = []
    for i in range(0, len(data), 2):
        points_arr.append((round(data[i] * 99), round(data[i + 1] * 99)))
    return points_arr


def train_network_max_examples(_network, repeats_number, train_mode=1, accuracy=1.0 / 255):
    input_files = helper.get_all_filenames('Dataset/Training_set/Without_points/')
    points_files = helper.get_all_filenames('Dataset/Training_set/With_points/')
    inputs = []
    outputs = []
    for i in range(len(points_files)):
        inputs.append(
            normalize_input(pixelizer.get_all_points('Dataset/Training_set/Without_points/' + input_files[i])))
        outputs.append(
            normalize_output(pixelizer.get_keypoints('Dataset/Training_set/With_points/' + points_files[i])))
    for i in range(len(points_files)):
        if train_mode == 1:
            _network.train(inputs[i], outputs[i], repeats_number)
        else:
            _network.train_accuracy(inputs[i], outputs[i], accuracy)

    # for k in range(repeats_number):
    #     for i in range(len(points_files)):
    #         _network.train(inputs[i], outputs[i], 1)


def train_network_one_example(file_name, _network, repeats_number, train_mode=1, accuracy=1.0 / 255):
    input_data = normalize_input(pixelizer.get_all_points('Dataset/Training_set/Without_points/' + file_name + '.jpg'))
    output_data = normalize_output(pixelizer.get_keypoints('Dataset/Training_set/With_points/' + file_name + '.png'))
    if train_mode == 1:
        _network.train(input_data, output_data, repeats_number)
    else:
        _network.train_accuracy(input_data, output_data, accuracy)


def smart_train(_network, filenames):
    max_error = (1.0 / 255) ** 2
    inputs = []
    outputs = []
    for file_name in filenames:
        inputs.append(normalize_input(
            pixelizer.get_all_points('Dataset/Training_set/Without_points/' + file_name + '.jpg')))
        outputs.append(normalize_output(
            pixelizer.get_keypoints('Dataset/Training_set/With_points/' + file_name + '.png')))
    epoch_numbers = [5]
    errors = [5]
    while np.max(errors) > max_error:
        epoch_numbers.clear()
        errors.clear()
        for i in range(len(filenames)):
            epoch_numbers.append(_network.train_accuracy(inputs[i], outputs[i], 1.0 / 255))
        for i in range(len(filenames)):
            errors.append(_network.get_error(inputs[i], outputs[i]))


def guess(_network, filename):
    _points = raw_data_to_points(_network.get_answer(
        normalize_input(pixelizer.get_all_points('Dataset/Training_set/Without_points/' + filename))))
    pixelizer.set_keypoints(_points, 'Dataset/Training_set/Without_points_rgb/', filename,
                            'Dataset/Training_set/Results/')


def guess_all(_network, folder_path):
    filenames = helper.get_all_filenames(folder_path)
    for filename in filenames:
        guess(_network, filename)


if __name__ == '__main__':
    # convert_all_images(64)
    network = Network()
    # network.set_start_weights()
    network.load_weights()
    # train_network_max_examples(network, 10, 1)
    # filename = '3'
    # train_network_one_example(filename, network, 10, 2, 2.0 * 1 / 255)

    smart_train(network, ['1', '2', '3', '4'])

    network.save_weights()
    guess(network, '1.jpg')
    guess(network, '2.jpg')
    guess(network, '3.jpg')
    guess(network, '4.jpg')
    # guess_all(network, 'Dataset/Training_set/Without_points/')
