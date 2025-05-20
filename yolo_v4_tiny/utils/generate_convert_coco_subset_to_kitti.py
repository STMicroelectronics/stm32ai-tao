# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
import argparse
import os
import json
from tqdm import tqdm
import shutil

"""
Usage:
python generate_convert_coco_subset_to_kitti.py --source-image-dir ./train2017 --source-annotation-file ./annotations/instances_train2017.json --out-data-dir ./coco2017_person/train2017 --num-images 100 --categories-to-keep person
python generate_convert_coco_subset_to_kitti.py --source-image-dir ./val2017 --source-annotation-file ./annotations/instances_val2017.json --out-data-dir ./coco2017_person/val2017 --num-images 100 --categories-to-keep person

# if you need more classes to keep then simple add more comma seperated class names to the last parameter like  <--categories-to-keep person,bicycle>
"""

def main():
    parser = argparse.ArgumentParser(description='Create a subset for coco with specified classes')
    parser.add_argument("--source-image-dir", type=str, required=True)
    parser.add_argument("--source-annotation-file", type=str, required=True)
    parser.add_argument("--out-data-dir", type=str, required=True)
    parser.add_argument("--num-images", type=int, default=0, help="Number of images to keep; if 0, keep all images")
    parser.add_argument("--categories-to-keep", type=str, required=True, help="Comma-separated list of categories to keep")
    args = parser.parse_args()
    
    source_image_dir = args.source_image_dir
    source_annotation_file = args.source_annotation_file
    out_data_dir = args.out_data_dir
    num_images = args.num_images
    categories_to_keep = args.categories_to_keep.split(',')

    if not os.path.exists(source_image_dir):
        raise Exception("Download and extract coco train2017/val2017/test2017")

    if source_annotation_file:
        if not os.path.exists(os.path.join(source_annotation_file)):
            raise Exception("Download and extract coco annotations.zip")

    if source_annotation_file:
        # Create the images directory
        out_image_dir = os.path.join(out_data_dir, "images")
        os.makedirs(out_image_dir, exist_ok=True)
        
        # Create the annotations directory
        out_json_dir = os.path.join(out_data_dir, "annotations")
        os.makedirs(out_json_dir, exist_ok=True)
        
        # Create the kitti_annotations directory
        kitti_annotation_dir = os.path.join(out_data_dir, "kitti_annotations")
        os.makedirs(kitti_annotation_dir, exist_ok=True)

        # Define the output JSON path
        out_json_path = os.path.join(out_json_dir, source_annotation_file.replace('\\', '/').split("/")[-1])
        inp_json_dict = json.load(open(source_annotation_file))
        print("Loaded input annotations")

        out_json_dict = {}
        out_json_dict["images"] = []
        out_json_dict["annotations"] = []

        id_set = set()

        # Get category IDs for the specified categories
        category_ids_to_keep = {category['id'] for category in inp_json_dict['categories'] if category['name'] in categories_to_keep}

        # Filter images that contain the specified categories
        image_ids_to_keep = {annot["image_id"] for annot in inp_json_dict["annotations"] if annot["category_id"] in category_ids_to_keep}

        image_id_to_filename = {img_info["id"]: img_info["file_name"] for img_info in inp_json_dict["images"]}

        for img_info in tqdm(inp_json_dict["images"]):
            if img_info["id"] in image_ids_to_keep:
                out_json_dict["images"].append(img_info)
                src_file_name = os.path.join(source_image_dir, img_info["file_name"])
                shutil.copy(src=src_file_name, dst=out_image_dir)
                id_set.add(img_info["id"])
                if num_images > 0 and len(out_json_dict["images"]) >= num_images:
                    break

        for annot_info in tqdm(inp_json_dict["annotations"]):
            if annot_info["image_id"] in id_set and annot_info["category_id"] in category_ids_to_keep:
                annot_info["area"] = float(annot_info["bbox"][2] * annot_info["bbox"][3])
                if annot_info["bbox"][0] <= 0.0:
                    annot_info["bbox"][0] = 0.0
                if annot_info["bbox"][1] <= 0.0:
                    annot_info["bbox"][1] = 0.0
                if annot_info["area"] > 0.0:
                    out_json_dict["annotations"].append(annot_info)

        # Include only the specified categories in the categories list
        out_json_dict["categories"] = [category for category in inp_json_dict["categories"] if category["id"] in category_ids_to_keep]

        json_out_str = json.dumps(out_json_dict, indent=4)
        with open(out_json_path, "w") as json_out_file:
            json_out_file.write(json_out_str)

        # Convert annotations to KITTI format
        convert_to_kitti(out_json_dict, kitti_annotation_dir, image_id_to_filename)

def convert_to_kitti(json_dict, kitti_dir, image_id_to_filename):
    """
    Convert COCO annotations to KITTI format.
    """
    for annot in json_dict["annotations"]:
        image_id = annot["image_id"]
        bbox = annot["bbox"]
        category_id = annot["category_id"]
        # Assuming category_id corresponds to a category name
        category_name = next((cat["name"] for cat in json_dict["categories"] if cat["id"] == category_id), "unknown")
        
        # KITTI format: class, x1, y1, x2, y2
        kitti_line = f"{category_name} 0.0 0.0 0.0 {bbox[0]} {bbox[1]} {bbox[0] + bbox[2]} {bbox[1] + bbox[3]} 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n"
        
        # Use the original file name for the KITTI annotation file
        kitti_file_name = image_id_to_filename[image_id].split('.')[0] + ".txt"
        kitti_file_path = os.path.join(kitti_dir, kitti_file_name)
        with open(kitti_file_path, "a") as kitti_file:
            kitti_file.write(kitti_line)

if __name__ == "__main__":
    main()