from glob import glob

def get_path(path):
    glob_file = glob(path + '/*')
    return glob_file
