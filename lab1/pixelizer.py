from PIL import Image
from os import listdir
from os.path import isfile, join


def get_pixels():
    return Image.open('Dataset/With_points/1.jpeg').load()


def count_points():
    total_points = 0
    img = Image.open('Dataset/Training_set/With_points/1.png', 'r')
    pixel_values = list(img.getdata())
    for i in pixel_values:
        if i[0] == 237 and i[1] == 28 and i[2] == 36:
            total_points += 1
    print(total_points)


def convert_image(filename):
    Image.open(f'Dataset/Training_set/Default_set/{filename}').resize((100, 100)).convert("L").save(
        f'Dataset/Training_set/Without_points/{filename}')


def get_all_filenames(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]


def convert_images(filenames, number):
    for i in range(number):
        convert_image(filenames[i])
