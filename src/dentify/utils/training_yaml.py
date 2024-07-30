import yaml
import json
from pathlib import Path

from ultralytics.data.utils import autosplit


def generate(coco_json_path, split_ratio=(0.9, 0.08, 0.02)):
    images_dir = Path(coco_json_path).parent / 'images'

    with open(coco_json_path, 'r') as file:
        coco = json.load(file)

    classes = {}
    for cat in coco['categories']:
        classes[cat['id']] = cat['name']

    data = dict(
        path=str(Path(coco_json_path).parent),
        train='autosplit_train.txt',
        val='autosplit_val.txt',
        test='autosplit_test.txt',

        names=classes
    )

    output_yaml = str(Path(coco_json_path).parent / 'data.yaml')
    with open(output_yaml, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False, sort_keys=False)

    autosplit(path=images_dir, weights=split_ratio, annotated_only=False)

    return output_yaml
