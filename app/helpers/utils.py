import os

# Get the app directory (parent of helpers directory)
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_data_path(*path_parts):
    """Get absolute path to data files relative to the app directory"""
    return os.path.join(APP_DIR, "data", *path_parts)
