import pixelizer


def convert_all_images(number):
    pixelizer.convert_images(pixelizer.get_all_filenames('Dataset/Training_set/Default_set/'), number)


if __name__ == '__main__':
    #convert_all_images(65)
    points = pixelizer.get_keypoints()
    pixelizer.set_keypoints(points, 'Dataset/Training_set/Without_points_rgb', '1.jpg', 'Dataset/Training_set/Trash')
