import pixelizer


def convert_all_images(number):
    pixelizer.convert_images(pixelizer.get_all_filenames('Dataset/Training_set/Default_set/'), number)


if __name__ == '__main__':
    # convert_all_images(65)
    print()
    pixelizer.count_points()
