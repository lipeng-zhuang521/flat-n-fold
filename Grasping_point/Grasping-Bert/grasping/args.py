import os
import numpy as np
import yaml


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class Arguments:
    def __init__(self, config_path, filename='default.yaml'):
        self.filename = os.path.splitext(filename)[0]
        with open(os.path.join(config_path, filename), 'r') as f:
            config = yaml.safe_load(f)

        for key, value in config.items():
            if isinstance(value, dict):
                setattr(self, key, Struct(**value))
            else:
                setattr(self, key, value)

        self.json = config
