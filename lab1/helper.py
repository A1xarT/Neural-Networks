from os import listdir
from os.path import isfile, join


def get_all_filenames(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]
