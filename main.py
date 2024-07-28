import os
import yaml
import argparse
from pathlib import Path
from types import SimpleNamespace

from ultralytics import YOLO

from src.dentify.utils.coco import convert_coco_json
from src.dentify.utils.training_yaml import generate


class ProgramArguments(object):
    def __init__(self):
        self.configs_path = None


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    return d


def main():
    args = parse_args()

    if args.configs_path is None:
        args.configs_path = r'configs/train_configs.yaml'

    if os.path.exists(args.configs_path):
        with open(args.configs_path, 'r') as conf_file:
            config = yaml.safe_load(conf_file)
        config = dict_to_namespace(config)
        dataset_path = config.data.coco_path

        # convert coco to yolo specific annotation format
        convert_coco_json(
            dataset_path,  # directory with *.json coco file
            use_segments=True,
        )

        # generate training yaml file and split dataset to train/val/test
        generate(
            dataset_path,
            split_ratio=(0.9, 0.08, 0.02)  # 90% training, 8% validation, 2% testing
        )

        # train the model


def parse_args():
    parser = argparse.ArgumentParser(description="Python package for CoTreat Challenge.")
    parser.add_argument("--configs_path", help="Path to the training configuration file."
                                               "default: `dentify.configs.train_configs.yaml`")
    program_arguments = ProgramArguments()
    parser.parse_args(namespace=program_arguments)

    return program_arguments


if __name__ == '__main__':
    main()
