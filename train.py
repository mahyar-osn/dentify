import os
import yaml
import argparse
from pathlib import Path
from types import SimpleNamespace

from ultralytics import YOLO

from src.dentify.coco import convert_coco_json
from src.dentify.training_yaml import generate


class ProgramArguments(object):
    def __init__(self):
        self.configs_path = None
        self.prepare_data = False


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

        if args.prepare_data:
            # convert coco to yolo specific annotation format
            convert_coco_json(
                dataset_path,  # directory with *.json coco file
                use_segments=True,
            )

            # generate training yaml file and split dataset to train/val/test
            _ = generate(
                dataset_path,
                split_ratio=(0.9, 0.08, 0.02)  # 90% training, 8% validation, 2% testing
            )

        # Note: we can uncomment the below two lines to log the training into Comet ML platform.
        # This was used for the challenge; however, it is not necessary.
        # import comet_ml
        # comet_ml.init(project_name=f"{config.experiment_name}")

        # train the model
        model = YOLO(config.model)
        model.train(data=str(Path(dataset_path).parent / 'data.yaml'),
                    name=config.experiment_name,
                    val=True,
                    epochs=config.hyperparams.epochs,
                    save_period=10,
                    imgsz=config.hyperparams.imgsz,
                    batch=config.hyperparams.batch,
                    device=config.hyperparams.device,
                    patience=config.hyperparams.patience,
                    lr0=config.hyperparams.lr0,
                    optimizer=config.hyperparams.optimizer,
                    weight_decay=config.hyperparams.weight_decay,
                    augment=True,
                    hsv_h=config.augment.hsv_h,
                    hsv_s=config.augment.hsv_s,
                    hsv_v=config.augment.hsv_v,
                    degrees=config.augment.degrees,
                    translate=config.augment.translate,
                    scale=config.augment.scale,
                    shear=config.augment.shear,
                    perspective=config.augment.perspective,
                    flipud=config.augment.flipud,
                    fliplr=config.augment.fliplr,
                    mosaic=config.augment.mosaic,
                    mixup=config.augment.mixup,
                    copy_paste=config.augment.copy_paste,
                    )


def parse_args():
    parser = argparse.ArgumentParser(description="Python package for CoTreat Challenge.")
    parser.add_argument("--configs_path", help="Path to the training configuration file."
                                               "default: `dentify.configs.train_configs.yaml`")
    parser.add_argument("--prepare_data", action='store_true', help="If set, prepare the data by converting COCO "
                                                                    "annotations and generating training splits.")

    program_arguments = ProgramArguments()
    parser.parse_args(namespace=program_arguments)

    return program_arguments


if __name__ == '__main__':
    main()
