import argparse
from typing import List

from ultralytics import YOLO


class ProgramArguments(object):
    def __init__(self):
        self.model_path = None
        self.image_paths = None


def main():
    args = parse_args()

    if args.model_path is None:
        raise FileNotFoundError(f"No model path is defined.")

    model = YOLO(args.model_path)
    if isinstance(args.image_paths, str):
        model(args.image_paths, save=True)
    elif isinstance(args.image_paths, List):
        for im in args.image_paths:
            model(im, save=True)
    else:
        raise TypeError(f"Invalid image paths {args.image_paths}. Either a string or list is possible.")


def parse_args():
    parser = argparse.ArgumentParser(description="Prediction module for the CoTreatAI Challenge.")
    parser.add_argument("--model_path", help="Path to the trained mode.")
    parser.add_argument("--image_paths", help="Path(s) to the image to be predicted. It can be a string pointing to a"
                                              "single image or a list of strings for multiple images.")

    program_arguments = ProgramArguments()
    parser.parse_args(namespace=program_arguments)

    return program_arguments


if __name__ == '__main__':
    main()
