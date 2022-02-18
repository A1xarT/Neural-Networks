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


def train_network_max_examples(_network, repeats_number):
    for k in range(repeats_number):
        points_files = helper.get_all_filenames('Dataset/Training_set/With_points/')
        input_files = helper.get_all_filenames('Dataset/Training_set/Without_points/')
        for i in range(len(points_files)):
            train_data_output = normalize_output(
                pixelizer.get_keypoints('Dataset/Training_set/With_points/' + points_files[i]))
            train_data_input = normalize_input(
                pixelizer.get_all_points('Dataset/Training_set/Without_points/' + input_files[i]))
            _network.train(train_data_input, train_data_output, 1)


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
    # convert_all_images(65)
    network = Network()
    network.set_start_weights()
    #network.load_weights()
    train_network_max_examples(network, 1000)
    network.save_weights()
    guess_all(network, 'Dataset/Training_set/Without_points/')
    # points = raw_data_to_points(network.get_answer(train_data_input))
    # pixelizer.set_keypoints(points, 'Dataset/Training_set/Without_points_rgb', '29_0_0_20170104165110771.jpg',
    #                       'Dataset/Training_set/Trash')
