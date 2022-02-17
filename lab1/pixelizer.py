from PIL import Image
from os import listdir
from os.path import isfile, join
import os


def get_keypoints():
    points = []
    keypoint_color = (237, 28, 36, 255)
    img = Image.open('Dataset/Training_set/With_points/1.png')
    pix = img.load()
    size = 100
    for j in range(size):
        for i in range(size):
            if pix[i, j] == keypoint_color:
                points.append((i, j))
    return points


def set_keypoints(points, folder_name, img_name, new_folder_name):
    keypoint_color = (237, 28, 36, 255)
    img = Image.open(folder_name + '/' + img_name)
    pix = img.load()
    for i in points:
        pix[i[0], i[1]] = keypoint_color
    # img.show()
    img.save(new_folder_name + '/' + os.path.splitext(img_name)[0] + '.png')


def convert_image(filename):
    img = Image.open(f'Dataset/Training_set/Default_set/{filename}').resize((100, 100))
    img.save(f'Dataset/Training_set/Without_points_rgb/{filename}')
    img.convert("L").save(f'Dataset/Training_set/Without_points/{filename}')


def get_all_filenames(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]


def convert_images(filenames, number):
    for i in range(number):
        convert_image(filenames[i])
