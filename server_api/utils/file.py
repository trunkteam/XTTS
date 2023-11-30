import os


def remove_if_exist(path):
    try:
        if path and path.strip() != "":
            os.remove(path)
    except OSError:
        pass
