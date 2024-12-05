import os

import io
import json
from collections import Counter
import re
import torch
import torch.nn.functional as F
import argparse
from PIL import Image
from torchvision import transforms

from segment_anything import sam_model_registry

from visualizer import Visualizer
from detectron2.data import MetadataCatalog
metadata = MetadataCatalog.get('coco_2017_train_panoptic')

from scipy.ndimage import label
import numpy as np

from gpt4v import request_gpt4v

import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors

from segment_anything import SamAutomaticMaskGenerator, SamPredictor

def get_bounding_boxes(canvas_result, image_ori):
    shpae_user = canvas_result.image_data.shape
    shape_original = image_ori.shape
    
    y_mul = shape_original[0]/shpae_user[0]
    x_mul = shape_original[1]/shpae_user[1]

    box_cor = []
    for data in canvas_result.json_data["objects"]:
        x_start = round(data['left']*x_mul)
        y_start = round(data['top']*y_mul)
        width = round(data['width']*x_mul)
        height = round(data['height']*y_mul)
        x_end = x_start + width
        y_end = y_start + height
        
        box_coordinates = [x_start, y_start, x_end, y_end]
        box_cor.append(box_coordinates)
        
    return box_cor

def gen_interacive_masks(image, bounding_boxes, model_sam):  
    model_sam.to(device='cuda')
    predictor = SamPredictor(model_sam)#this creates a SamPredictor object using the sam model
    predictor.set_image(image)

    masks_interactive = []

    for box_cor in bounding_boxes:
        input_box = np.array(box_cor)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        masks = np.squeeze(masks, axis=0)
        masks_interactive.append(masks)
    return masks_interactive

def filter_masks(mask_interactive, mask_all, image_ori):
    mask = mask_interactive.copy()

    # --------------------Remove overlapped masks--------------------------##
    mask_rm_in = []
    mask_all_new = []
    
    for ref_index, ref_mask in enumerate(mask_all):
        if ref_mask['area']/(image_ori.shape[0]*image_ori.shape[1]) < 0.001:
            mask_rm_in.append(ref_index)
    for index in range(len(mask_all)):
        if index not in mask_rm_in:
            mask_all_new.append(mask_all[index])
        # del mask_all[index]
    mask_all = mask_all_new.copy()
    mask_all_main = mask_all.copy()
    
    for ref_index, ref_data in enumerate(mask_all_main):
        ref_mask = ref_data['segmentation']
        mask_rm_in = []
        mask_all_new = []
        for tmp_index, test_data in enumerate(mask_all):
            if tmp_index == ref_index:
                continue
        
            test_mask = test_data['segmentation']
            intersection = np.logical_and(ref_mask, test_mask)
            union = np.logical_or(ref_mask, test_mask)
            
            union = np.sum(union)
            intersection = np.sum(intersection)
            iou = intersection/union
            
            if iou > 0.001:
                if ref_data["area"] > test_data["area"]:
                    mask_rm_in.append(tmp_index)
                else:
                    mask_rm_in.append(ref_index)
                
        for index in range(len(mask_all)):
            if index not in mask_rm_in:
                mask_all_new.append(mask_all[index])
            # del mask_all[index]
        mask_all = mask_all_new.copy()
    
    mask_rm_in = []
    mask_all_new = []

    # ---------------------Remove overlapped mask with the main interactive masks-------------------###
    for ref_mask in mask:
        # ref_mask = ref_mask['segmentation']
        for test_index, test_mask in enumerate(mask_all):
            test_mask = test_mask['segmentation']
            intersection = np.logical_and(ref_mask, test_mask)
            union = np.logical_or(ref_mask, test_mask)
            
            union = np.sum(union)
            intersection = np.sum(intersection)
            iou = intersection/union
            
            if iou > 0.01:
                mask_rm_in.append(test_index)
                
    for index in range(len(mask_all)):
        if index not in mask_rm_in:
            mask_all_new.append(mask_all[index])
        # del mask_all[index]
    mask_all = mask_all_new.copy()

    return mask_all
    
def generate_gpt4_in_image(image, masks_interactive, mask_all):
    visual = Visualizer(image, metadata = metadata)
    label = 1  
    
    alpha = 0.5
    label_mode = '1'
    
    for i, mask in enumerate(masks_interactive):
        demo = visual.draw_binary_mask_with_number(mask, text=str(label), label_mode=label_mode, alpha=0.3, anno_mode="Mask")
        demo = visual.draw_binary_mask_with_number(mask, text=str(label), label_mode=label_mode, alpha=alpha, anno_mode="Mark")
        label += 1
    
    for i, mask in enumerate(mask_all):
        mask = mask['segmentation']
        
        demo = visual.draw_binary_mask_with_number(mask, text=str(label), label_mode=label_mode, alpha=alpha, anno_mode="Mark")
        label += 1
        
    image = demo.get_image()    

    return image

def generate_out_image(image, masks_interactive, mask_all, dict_class, user_res):
    pattern = r'\[(.*?)\]'
    matches = list(set(re.findall(pattern, user_res)))
    visual = Visualizer(image, metadata = metadata)
    
    alpha = 0.5
    label_mode = '1'

    label = 1  
    for i, mask in enumerate(masks_interactive):
        text = dict_class[str(label)] #+f"\nObject_{label}"
        demo = visual.draw_binary_mask_with_number(mask, text=str(1), label_mode=label_mode, alpha=alpha, anno_mode="Mask")
        demo = visual.draw_binary_mask_with_number(mask, text=text, label_mode=label_mode, alpha=alpha, anno_mode="Mark")
        label += 1
    
    for i, mask in enumerate(mask_all):
        if str(label) in matches:
            text = dict_class[str(label)] #+f"\nObject_{label}"
            mask = mask['segmentation']
            demo = visual.draw_binary_mask_with_number(mask, text=text, label_mode=label_mode, alpha=alpha, anno_mode="Mask")
            demo = visual.draw_binary_mask_with_number(mask, text=text, label_mode=label_mode, alpha=alpha, anno_mode="Mark")  
        label += 1
        
    image = demo.get_image()  
    return image

def gen_mask_all(model, image_ori):
    mask_generator = SamAutomaticMaskGenerator(model)
    outputs = mask_generator.generate(image_ori)

    # sorted masks based on area
    sorted_masks = sorted(outputs, key=(lambda x: x['area']), reverse=True)
    
    return sorted_masks

def gen_dict(sysyem_response):
    pattern = r'\{(.*?)\}'
    matches = re.findall(pattern, sysyem_response)
    dict_stirng = "{"+matches[0]+"}"
    dict_out = json.loads(dict_stirng)
    return dict_out

def replace_marks(string, replacements):
    # Regular expression to match custom placeholders
    pattern = r'\[([^\[\]]*)\]'
    return re.sub(pattern, lambda match: replacements.get(match.group(1), match.group(0)), string)

def filter_response(response):
    all_res = response.rsplit('++Answer++', 1)
    
    system_res = all_res[0].replace("\n", '').strip()
    user_res = all_res[1].replace("\n", '').strip()
    
    dict_class = gen_dict(system_res)
    
    values = [data[1] for data in dict_class.items()]
    freq = dict(Counter(values))
    
    for value, frequency in freq.items():
        if frequency > 1:
            count = 1
            for key, class_name in dict_class.items():
                if class_name == value:
                    class_name = f"{class_name}_{count}"
                    dict_class[key] = class_name
                    count += 1
    
    user_res_filter = replace_marks(user_res, dict_class)
    user_res_filter = re.sub(r'\b(\w+)\s+\1\b', r'\1', user_res_filter)
    
    return user_res_filter, dict_class, user_res

def gpt_4_response(image_ori, user_message, masks_interactive, mask_all):
    number_obj_user = ', '.join(map(str, range(1,len(masks_interactive)+1)))
    number_obj = ', '.join(map(str, range(1,(len(mask_all)+len(masks_interactive)+1))))
            
    sys_msg_1 = f"All the objects in the image are marked using numbers {number_obj}. Sequentially the user selected objects are {number_obj_user}. User selected objects are also segmented."
    sys_msg_2 = """First please provide the name of the objects properly, which are marked using numbers. Use the following format: class_dict = {"marked number":"object name"}. Then answer to the question. For any marks mentioned in your answer, please highlight them with []. Please don't give both [mark] and object name. Avoid this type of sentences "[mark] is object name". Start answer with the symbol "++Answer++". Please provide "class_dict" before "++Answer++".""" 
    sys_msg_3 = f"\n"
    
    system_message = sys_msg_1+" "+sys_msg_2+" "+sys_msg_3

    image = generate_gpt4_in_image(image_ori, masks_interactive, mask_all)
    image = Image.fromarray(image)
    res = ""
    while not (res.find("++Answer++") != -1 or res.find("class_dict") != -1):
        res = request_gpt4v(system_message, user_message, image)

    res_filteres, dict_class, user_res = filter_response(res)
    out_image = generate_out_image(image_ori, masks_interactive, mask_all, dict_class, user_res)
    
    return res_filteres, out_image, dict_class, res

def intelligent_eye(image_in, canvas_result, user_question):
    sam_ckpt = "sam_vit_h_4b8939.pth"
    model_sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).eval().cuda()
    
    image_ori = np.array(image_in)
    image_ori = image_ori[...,:3]
    
    bounding_boxes = get_bounding_boxes(canvas_result, image_ori)

    with torch.autocast(device_type='cuda', dtype=torch.float16):
        masks_interactive = gen_interacive_masks(image_ori, bounding_boxes, model_sam)
        mask_all = gen_mask_all(model_sam, image_ori)
        mask_all = filter_masks(masks_interactive, mask_all, image_ori)
        
    res_filteres, out_image, dict_class, res = gpt_4_response(image_ori, user_question, masks_interactive, mask_all)

    return out_image, res_filteres
