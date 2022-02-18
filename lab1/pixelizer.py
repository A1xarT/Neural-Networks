from PIL import Image
import os


def get_keypoints(path):
    points = []
    keypoint_color = (237, 28, 36, 255)
    img = Image.open(path)
    pix = img.load()
    size = 100
    for j in range(size):
        for i in range(size):
            if pix[i, j] == keypoint_color:
                points.append(i)
                points.append(j)
    return points


def set_keypoints(points, folder_name, img_name, new_folder_name):
    keypoint_color = (237, 28, 36, 255)
    img = Image.open(folder_name + img_name)
    pix = img.load()
    for i in points:
        pix[i[0], i[1]] = keypoint_color
    img.save(new_folder_name + os.path.splitext(img_name)[0] + '.png')


def convert_image(filename):
    img = Image.open(f'Dataset/Training_set/Default_set/{filename}').resize((100, 100))
    img.save(f'Dataset/Training_set/Without_points_rgb/{filename}')
    img.convert("L").save(f'Dataset/Training_set/Without_points/{filename}')


def convert_images(filenames, number):
    for i in range(number):
        convert_image(filenames[i])


def get_all_points(path):
    img = Image.open(path, 'r').load()
    arr = []
    for j in range(100):
        for i in range(100):
            arr.append(img[i, j])
    return arr
