import subprocess
import cv2
import json
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import argparse
import time
from pathlib import Path

import json
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

import os
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

def detect_function(source, weights, name,img_size=640, conf_thres=0.25, iou_thres=0.45, device='', view_img=False,
                    save_txt=False, save_conf=False, nosave=False, classes=None, agnostic_nms=False,
                    augment=False, update=False, project='runs/detect',  exist_ok=False, no_trace=False):
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.source = source
    opt.weights = weights
    opt.img_size = img_size
    opt.conf_thres = conf_thres
    opt.iou_thres = iou_thres
    opt.device = device
    opt.view_img = view_img
    opt.save_txt = save_txt
    opt.save_conf = save_conf
    opt.nosave = nosave
    opt.classes = classes
    opt.agnostic_nms = agnostic_nms
    opt.augment = augment
    opt.update = update
    opt.project = project
    opt.name = name
    opt.exist_ok = exist_ok
    opt.no_trace = no_trace
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    # source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    # image_basename = os.path.splitext(os.path.basename(source))[0]
    # save_dir = Path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # save_dir = Path(opt.project) / opt.name / "result" / image_basename
    # (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # img.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                result = {
                # "directory": str(save_dir),
                "path": str(source),
                "prediction":[],
                
                }
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
              
                for *xyxy, conf, cls in reversed(det):
                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                    #     with open(txt_path + '.txt', 'a') as f:
                    #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                    prediction_info = {
                            "class": int(cls),
                            "label": names[int(cls)],
                            "confidence": float(conf),
                            "bounding_box": [float(coord) for coord in xyxy],
                            # "img": im0
                        }
                    result["img"] = im0
                    result["prediction"].append(prediction_info)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

    print(f'Done. ({time.time() - t0:.3f}s)')
    return result

def detect(image_path, image_name):
    result = detect_function(image_path,'./best.pt', image_name)
    return result
    # detection_command = f"python ./yolov7/detect.py --weights ./yolov7/best.pt --conf 0.1 --source {image_path} --name {image_name}"
    # result = subprocess.run(detection_command, shell=True, text=True, capture_output=True)

def predict_and_display(image, model):
    img_array = preprocess_image(image)
    CATEGORIES = ['Black', 'Blue', 'Brown', 'Gray', 'Green', 'Orange', 'Pink', 'Purple', 'Red', 'White', 'Yellow']
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return CATEGORIES[predicted_class]

def preprocess_image(image_array, target_size=(32,32)):
    resized_image = cv2.resize(image_array, (target_size[1], target_size[0]))
    img_array = np.expand_dims(resized_image, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

def process_image(image_path):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    segmentation_result = detect(image_path, image_name)
    # print(segmentation_result)
    # json_path = f'./runs/detect/{image_name}/result.json'

    # with open(json_path, 'r') as json_file:
    #     data = json.load(json_file)

    image_segmented_path = segmentation_result['path']
    # Testing to see segmentation img
    # The segmentation image is in segmentation_result["img"]
    # os.makedirs(f'./result', exist_ok=True)
    # save_path = f'./result/result.jpg'
    # cv2.imwrite(save_path, segmentation_result["img"])


    image = cv2.imread(image_path)
    highest_confidence_per_label = {}
    for prediction in segmentation_result['prediction']:
        bounding_box = prediction['bounding_box']
        label = prediction['label']
        confidence = prediction['confidence']

        bounding_box = [int(coord) for coord in bounding_box]
        if label not in highest_confidence_per_label or confidence > highest_confidence_per_label[label]['confidence']:
            highest_confidence_per_label[label] = {
                'bounding_box': bounding_box,
                'label': label,
                'confidence': confidence
            }
    
    result_prediction = []
    pred = {
            image_name:{
                
            }
            ,
            "path": image_segmented_path,
            "segmented_image": segmentation_result["img"]
        }
    cropped_image =[]
    for label, highest_confidence_prediction in highest_confidence_per_label.items():
        bounding_box = highest_confidence_prediction['bounding_box']
        label = highest_confidence_prediction['label']
        confidence = highest_confidence_prediction['confidence']

        # Crop the region of interest (ROI) using the bounding box
        cropped_roi = image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]

        # Save the cropped ROI to a file
        cropped_by_label = {}
        cropped_by_label[label] = cropped_roi
        cropped_image.append(cropped_by_label)
        rgb_image = cv2.cvtColor(cropped_roi, cv2.COLOR_BGR2RGB)
        color_model = load_model('./color_classification_cnn_model.h5')
        categories = predict_and_display(rgb_image, color_model)

        convert_putih = ["Pink","Cream","Gray","Red","Yellow"]
        convert_brown = ["Purple","Orange","Green","Blue"]
        if label == "skin":
            if categories in convert_putih:
                categories = "White"
            elif categories in convert_brown:
                categories = "Brown"

        pred[image_name][label] = categories
    result_prediction.append(pred)

    return result_prediction

result = process_image('./0003.jpg')
print(result)