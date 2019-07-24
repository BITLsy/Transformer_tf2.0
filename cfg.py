import os
import yaml


BASE_DIR = os.path.dirname(__file__)


def get_path(relative_path):
    path = os.path.join(BASE_DIR, relative_path)
    return path


with open(get_path('config.yaml')) as fin:
    CONFIG = yaml.safe_load(fin)


