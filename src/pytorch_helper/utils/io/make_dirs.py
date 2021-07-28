import os


def make_dirs(path: str):
    """ create directories of `path`

    :param path: str of the directory path to create
    """
    os.makedirs(os.path.abspath(path), exist_ok=True)


def make_dirs_for_file(path: str):
    """ create the folder for the file specified by `path`

    :param path: str of the file
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
