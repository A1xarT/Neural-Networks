import pixelizer
from normalization import normalize_output, normalize_input


def convert_all_images(number):
    pixelizer.convert_images(pixelizer.get_all_filenames('Dataset/Training_set/Default_set/'), number)


if __name__ == '__main__':
    # convert_all_images(65)
    # pixelizer.set_keypoints(points, 'Dataset/Training_set/Without_points_rgb', '1.jpg', 'Dataset/Training_set/Trash')
    train_data_output = normalize_output(pixelizer.get_keypoints())
    train_data_input = normalize_input(pixelizer.get_all_points('Dataset/Training_set/Without_points/1.jpg'))
    print(train_data_output)
