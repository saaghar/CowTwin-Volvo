#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main testing inference script.

"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import time

import logging
import argparse
from collections import OrderedDict
import torch
import cv2
from PIL import Image
import json

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Instances
from detectron2.layers import batched_nms

import torch.multiprocessing
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')


import lib.data.fewshot
import lib.data.ovdshot
from lib.categories import SEEN_CLS_DICT, ALL_CLS_DICT

from collections import defaultdict
from tqdm import tqdm

import numpy as np

from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.data.datasets.coco_zeroshot_categories import COCO_SEEN_CLS, \
    COCO_UNSEEN_CLS, COCO_OVD_ALL_CLS
    
from sklearn.metrics import precision_recall_curve
from sklearn import metrics as sk_metrics

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []

        if 'OpenSet' in cfg.MODEL.META_ARCHITECTURE:
            if 'lvis' in dataset_name:
                evaluator_list.append(LVISEvaluator(dataset_name, output_dir=output_folder))
            else:
                dtrain_name = cfg.DATASETS.TRAIN[0]
                # for coco14 FSOD benchmark
                if 'coco' in dataset_name:
                    seen_cnames = SEEN_CLS_DICT['fs_coco14_base_train']
                else:
                    seen_cnames = SEEN_CLS_DICT[dtrain_name]
                all_cnames = ALL_CLS_DICT[dtrain_name]
                unseen_cnames = [c for c in all_cnames if c not in seen_cnames]
                evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder, few_shot_mode=True,
                                                seen_cnames=seen_cnames, unseen_cnames=unseen_cnames,
                                                all_cnames=all_cnames))
            return DatasetEvaluators(evaluator_list)
    

        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

def run_inference_on_image(image_path, predictor, cfg, verbose, save_img, threshold=0.6, output_path="result.jpg", nms_threshold=0.5):
    """
    Run inference on a single image and save the visualization.
    
    Args:
        image_path (str): Path to the input image.
        predictor (DefaultPredictor): Predictor object.
        cfg (CfgNode): Config object.
        threshold (float): Score threshold for detections.
        output_path (str): Path to save the output visualization.
        iou_threshold (float): IoU threshold for NMS to filter redundant boxes.
        
    Returns:
        dict: Prediction results.
    """
    total_start_time = time.time()
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Run inference
    inference_start_time = time.time()
    outputs = predictor(image)
    inference_time = time.time() - inference_start_time
    
    dataset_name = cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) > 0 else "__unused"
    metadata = MetadataCatalog.get(dataset_name)

    instances = outputs["instances"].to("cpu")
    
    # Apply score threshold
    score_filter = instances.scores >= threshold
    filtered_instances = instances[score_filter]
    
    if len(filtered_instances) > 0:
        boxes_t = filtered_instances.pred_boxes.tensor
        scores_t = filtered_instances.scores
        classes_t = filtered_instances.pred_classes
        
        # Apply batched NMS (per-class NMS)
        keep = batched_nms(boxes_t, scores_t, classes_t, nms_threshold)
        
        # Create new instances with only the kept boxes
        final_instances = filtered_instances[keep]
    else:
        final_instances = filtered_instances
            
    # Visualize results
    v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.2)
    v = v.draw_instance_predictions(final_instances)

    result_image = v.get_image()[:, :, ::-1]
    if save_img:
        cv2.imwrite(output_path, result_image)
    
    total_time = time.time() - total_start_time

    if len(filtered_instances) > 0:
        classes = final_instances.pred_classes.tolist()
        scores = final_instances.scores.tolist()
        boxes = final_instances.pred_boxes.tensor.tolist()
    else:
        classes = []
        scores = []
        boxes = []

    img_name = os.path.basename(image_path)
    #print(f"Image name is: {img_name}")
    img_height, img_width, _ = image.shape

    class_tups = []
    
    for i in range(len(classes)):
        class_id = classes[i]
        class_name = metadata.thing_classes[class_id] if hasattr(metadata, "thing_classes") else f"Class {class_id}"
        class_tups.append((class_name, class_id, tuple(boxes[i])))
        
    predictions = {img_name: [img_width, img_height, class_tups]}

    if verbose:
        print(f"Result saved to {output_path}")
        print(f"Inference time: {inference_time:.4f} seconds")
        print(f"Total processing time: {total_time:.4f} seconds")
        print(f"Detected {len(final_instances)} objects with score >= {threshold}")

        if len(filtered_instances) > 0:
            for i in range(len(final_instances)):
                class_id = classes[i]
                class_name = metadata.thing_classes[class_id] if hasattr(metadata, "thing_classes") else f"Class {class_id}"
                print(f"Object {i+1}: {class_name}, Score {scores[i]:.4f}, Box {boxes[i]}")
    
    return outputs, predictions

def to_coco(image_annotations: dict[str, list[int, int, list[tuple[str, int, tuple[int, int, int, int]]]]]):
    coco = {
        "images": [],
        "annotations": [],
    }

    i_anno = 0 
    for i_im, img_name in enumerate(image_annotations):
        img_prop = image_annotations[img_name]
        image = {
            "id": i_im,
            "width": img_prop[0],
            "height": img_prop[1],
            "file_name": img_name
        }

        coco["images"].append(image)
        
        for annotation in img_prop[2]:
            label, category_id, coord = annotation
            x1, y1, x2, y2 = coord
            w = x2 - x1
            h = y2 - y1
            coco["annotations"].append({
                "id": i_anno,
                "image_id": i_im,
                "category_id": category_id,
                "bbox": [x1, y1, w, h]
            })
            i_anno += 1

    return coco
    
def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DE.CONTROLLER = args.controller

    cfg.freeze()
    default_setup(cfg, args)
    print(cfg.DATASETS.TEST)
    return cfg


def modified_argument_parser():
    """
    Create a parser with additional arguments for single image inference.
    """
    parser = default_argument_parser()
    parser.add_argument("--image-path", required=True, help="Path to the image or folder of images for inference")
    parser.add_argument("--output-path", required=True, help="Path to save the output image")
    parser.add_argument("--save-img", action="store_true", help="Enable storing labeled images in the output path")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection score threshold")
    parser.add_argument("--nms-threshold", type=float, default=0.2, help="IoU threshold for NMS")
    return parser


def main(args):
    cfg = setup(args)

    if not os.path.isdir(args.image_path):
        print("Image folder path is not a directory")
        return
    else:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )

        verbose = args.verbose
        save_img = args.save_img
    
        # Create predictor
        predictor = DefaultPredictor(cfg)
        
        print(f"Detected folder path: {args.image_path}")

        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
            
        # Get all image files in the folder
        image_files = []
        for filename in os.listdir(args.image_path):
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                image_files.append(os.path.join(args.image_path, filename))
            
        if not image_files:
            print(f"No image files found in {args.image_path}")
            return {}
                
        print(f"Found {len(image_files)} images to process")

        # Clear output path directory
        print(f"Directory cleared: {args.output_path}")
        for item in os.listdir(args.output_path):
            item_path = os.path.join(args.output_path, item)
        
            if os.path.isfile(item_path):
                os.remove(item_path)
        
        # Process each image
        results = {}
        all_preds = {}
        for i, image_path in enumerate(tqdm(image_files, desc="Processing images")):
            img_name = os.path.basename(image_path)
            if verbose:
                print(f"\nProcessing image {i+1}/{len(image_files)}: {img_name}")
                    
            # Create output path
            output_filename = f"{os.path.splitext(img_name)[0]}_result.jpg"
            output_path = os.path.join(args.output_path, output_filename)
                
            try:
                # Run inference on the image
                results[img_name], preds = run_inference_on_image(
                    image_path, 
                    predictor, 
                    cfg, 
                    verbose,
                    save_img,
                    threshold=args.threshold, 
                    output_path=output_path,
                    nms_threshold=args.nms_threshold
                )

                all_preds.update(preds)
                
            except Exception as e:
                print(f"Error processing {img_name}: {e}")

        all_preds_coco = to_coco(all_preds)

        with open('datasets/trucks_full_dataset/annotations/1_shot.json', 'r') as file:
            data_10_shot = json.load(file)
        all_categories = data_10_shot['categories']
        
        all_preds_coco['categories'] = all_categories
        
        with open('distilled_labels_03_1shot.json', 'w') as fp:
            json.dump(all_preds_coco, fp)
    
        print(f"\nAll images processed. Results saved to: {args.output_path}")
        return results

if __name__ == "__main__":
    args = modified_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )