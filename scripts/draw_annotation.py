"""
This script is used to draw and label teeth in an image using a corresponding annotation text file.
I wrote this script mainly to visualize the annotations for the testing dataset to compare against the predictions.
"""
import cv2
import numpy as np


def read_image(image_path):
    """Read the image using OpenCV."""
    image = cv2.imread(image_path)
    return image


def read_annotation(annotation_path):
    """Read the YOLOv8 segmentation annotation from the TXT file."""
    with open(annotation_path, 'r') as file:
        lines = file.readlines()
    return lines


def parse_annotation(lines, image_shape):
    """Parse the annotation to extract the segmentation data."""
    height, width = image_shape[:2]
    masks = []

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        polygon = np.array([float(coord) for coord in parts[1:]]).reshape(-1, 2)
        polygon[:, 0] *= width
        polygon[:, 1] *= height
        masks.append((class_id, polygon.astype(int)))

    return masks


def draw_segmentation(image, masks, class_names):
    """Draw the segmentation boundary on the image and add class names."""
    for class_id, polygon in masks:
        cv2.polylines(image, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)
        class_name = class_names.get(class_id, f"Class {class_id}")
        x, y = polygon.mean(axis=0).astype(int)
        cv2.putText(image, class_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(image, class_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return image


def save_image(image, output_path):
    """Save the annotated image."""
    cv2.imwrite(output_path, image)


def main(image_path, annotation_path, output_path, class_names):
    image = read_image(image_path)
    lines = read_annotation(annotation_path)
    masks = parse_annotation(lines, image.shape)
    annotated_image = draw_segmentation(image, masks, class_names)
    save_image(annotated_image, output_path)


# Testing
if __name__ == "__main__":
    imgs = ['april2015-pbw-completed__ecd75fa1d25d8cd766177270387791b5',
            'march2015-pbw-completed__c5b2adaf93ba04f6355210b14f00134b',
            'november2015-pbws-complete__f4ee572aceba365948ccf7f9d4b7f33c',
            'pbws-super-set-1-completed__PBWs_Super_Set_1 - 0000000029',
            'pbws-super-set-1-completed__PBWs_Super_Set_1 - 0000000298',
            'pbws-super-set-1-completed__PBWs_Super_Set_1 - 0000000302',
            'pbws-super-set-1-completed__PBWs_Super_Set_1 - 0000000510',
            'pbws-super-set-1-completed__PBWs_Super_Set_1 - 0000000703',
            'pbws-super-set-1-completed__PBWs_Super_Set_1 - 0000000834',
            'pbws-super-set-1-completed__PBWs_Super_Set_1 - 0000000840',
            'pbws-super-set-1-completed__PBWs_Super_Set_2 - 0000000046',
            'pbws-super-set-1-completed__PBWs_Super_Set_2 - 0000000123',
            'pbws-super-set-1-completed__PBWs_Super_Set_2 - 0000000481',
            'pbws-super-set-1-completed__PBWs_Super_Set_2 - 0000000687',
            'pbws-super-set-1-completed__PBWs_Super_Set_2 - 0000000907',
            'pbws-super-set-1-completed__PBWs_Super_Set_3 - 0000000647',
            'september2015-pbws-completed__50ade4f36afa742f34395abfa5cdfdcb']

    class_names = {
        0: "tooth 15",
        1: "tooth 44",
        2: "tooth 17",
        3: "tooth 45",
        4: "tooth 47",
        5: "Calculus",
        6: "caries",
        7: "tooth 14",
        8: "composite",
        9: "tooth 46",
        10: "tooth 16",
        11: "tooth 26",
        12: "tooth 37",
        13: "tooth 34",
        14: "tooth 25",
        15: "tooth 27",
        16: "tooth 24",
        17: "tooth 28",
        18: "tooth 38",
        19: "tooth 36",
        20: "tooth 35",
        21: "crown",
        22: "root filling",
        23: "tooth 23",
        24: "tooth 33",
        25: "amalgam",
        26: "tooth 13",
        27: "tooth 43",
        28: "tooth 48",
        29: "tooth 18",
        30: "tooth 12",
        31: "tooth 32",
        32: "tooth 22",
        33: "implant",
        34: "periapical pathology"
    }

    for im in imgs:
        image_path = rf'../cotreat-computer-vision-ai-engineer-challenge-dataset-2024-mahyar/images/{im}.jpg'
        annotation_path = rf'../cotreat-computer-vision-ai-engineer-challenge-dataset-2024-mahyar/labels/{im}.txt'
        output_path = rf'../cotreat-computer-vision-ai-engineer-challenge-dataset-2024-mahyar/eval/{im}.jpg'
        main(image_path, annotation_path, output_path, class_names)
