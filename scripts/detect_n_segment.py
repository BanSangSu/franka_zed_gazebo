#!/usr/bin/env python3

"""
Multi-Detector + SAM Integrated ROS Node
Supports: Florence-2, DINO, Grounding DINO, YOLOv11 + SAM Segmentation
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import rospy
import torch
from PIL import Image

# SAM imports
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    rospy.logwarn("Warning: segment_anything not installed. Install with: pip install segment-anything")

# ==================== BASE DETECTOR INTERFACE ====================

class BaseDetector(ABC):
    """Abstract base class for all object detectors"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    @abstractmethod
    def detect(self, image: Image.Image, prompt: str = None, **kwargs) -> Dict:
        pass
    
    @abstractmethod
    def load_model(self):
        pass

# ==================== FLORENCE-2 DETECTOR ====================

class Florence2Detector(BaseDetector):
    def __init__(self, model_name: str = "microsoft/Florence-2-base"):
        super().__init__(model_name)
        self.load_model()
    
    def load_model(self):
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            rospy.loginfo(f"Loading Florence-2 ({self.model_name}) on {self.device}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=self.torch_dtype, trust_remote_code=True
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            rospy.loginfo("Florence-2 loaded successfully!")
        except Exception as e:
            rospy.logerr(f"Failed to load Florence-2: {e}")
            raise
    
    def detect(self, image: Image.Image, prompt: str = None, **kwargs) -> Dict:
        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        full_prompt = task_prompt + prompt if prompt else task_prompt
        
        inputs = self.processor(text=full_prompt, images=image, return_tensors="pt").to(self.device, self.torch_dtype)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
            max_new_tokens=1024, num_beams=3, do_sample=False
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(image.width, image.height)
        )
        grounding_data = parsed_answer.get(task_prompt, {})
        bboxes = grounding_data.get('bboxes', [])
        return {
            'bboxes': bboxes,
            'labels': grounding_data.get('labels', []),
            'scores': [1.0] * len(bboxes),
            'found': len(bboxes) > 0,
            'num_detections': len(bboxes),
            'prompt': prompt
        }

# ==================== GROUNDING DINO DETECTOR ====================

class GroundingDINODetector(BaseDetector):
    def __init__(self, model_name: str = "IDEA-Research/grounding-dino-base"):
        super().__init__(model_name)
        self.load_model()
    
    def load_model(self):
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            rospy.loginfo(f"Loading Grounding DINO ({self.model_name}) on {self.device}...")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_name).to(self.device)
            rospy.loginfo("Grounding DINO loaded successfully!")
        except Exception as e:
            rospy.logerr(f"Error loading Grounding DINO: {e}")
            raise
    
    def detect(self, image: Image.Image, prompt: str = None, box_threshold: float = 0.35, text_threshold: float = 0.25, **kwargs) -> Dict:
        if not prompt: prompt = "object"
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=box_threshold, text_threshold=text_threshold, target_sizes=[image.size[::-1]]
        )[0]
        return {
            'bboxes': results["boxes"].cpu().numpy().tolist(),
            'labels': results["labels"],
            'scores': results["scores"].cpu().numpy().tolist(),
            'found': len(results["boxes"]) > 0,
            'num_detections': len(results["boxes"]),
            'prompt': prompt
        }

# ==================== YOLOV11 DETECTOR ====================

class YOLOv11Detector(BaseDetector):
    def __init__(self, model_name: str = "yolo11n.pt"):
        super().__init__(model_name)
        self.load_model()
    
    def load_model(self):
        try:
            from ultralytics import YOLO
            rospy.loginfo(f"Loading YOLOv11 ({self.model_name}) on {self.device}...")
            self.model = YOLO(self.model_name)
            rospy.loginfo("YOLOv11 loaded successfully!")
        except Exception as e:
            rospy.logerr(f"Error loading YOLOv11: {e}")
            raise
    
    def detect(self, image: Image.Image, prompt: str = None, conf: float = 0.25, iou: float = 0.45, classes: List[int] = None, **kwargs) -> Dict:
        results = self.model(image, conf=conf, iou=iou, classes=classes, verbose=False)[0]
        boxes = results.boxes
        bboxes = boxes.xyxy.cpu().numpy().tolist()
        labels = [results.names[int(c)] for c in boxes.cls.cpu().numpy()]
        scores = boxes.conf.cpu().numpy().tolist()
        
        if prompt:
            filtered = [(b, l, s) for b, l, s in zip(bboxes, labels, scores) if prompt.lower() in l.lower()]
            bboxes, labels, scores = zip(*filtered) if filtered else ([], [], [])
        
        return {
            'bboxes': list(bboxes),
            'labels': list(labels),
            'scores': list(scores),
            'found': len(bboxes) > 0,
            'num_detections': len(bboxes),
            'prompt': prompt
        }

# ==================== DINO DETECTOR ====================

class DINODetector(BaseDetector):
    def __init__(self, model_name: str = "facebook/dinov2-base"):
        super().__init__(model_name)
        self.load_model()
    
    def load_model(self):
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
    
    def detect(self, image: Image.Image, prompt: str = None, **kwargs) -> Dict:
        rospy.logwarn("DINOv2 direct detection not supported. Use Grounding DINO.")
        return {'bboxes': [], 'labels': [], 'scores': [], 'found': False, 'num_detections': 0}

# ==================== UNIFIED PIPELINE ====================

class MultiDetectorSAM:
    def __init__(self, detector_type: str = "florence2", detector_config: Dict = None, 
                 sam_checkpoint: str = "sam_vit_h_4b8939.pth", sam_model_type: str = "vit_h"):
        self.detector_type = detector_type
        detector_config = detector_config or {}
        
        if detector_type == "florence2":
            self.detector = Florence2Detector(detector_config.get("model_name", "microsoft/Florence-2-base"))
        elif detector_type == "grounding_dino":
            self.detector = GroundingDINODetector(detector_config.get("model_name", "IDEA-Research/grounding-dino-base"))
        elif detector_type == "yolov11":
            self.detector = YOLOv11Detector(detector_config.get("model_name", "yolo11n.pt"))
        elif detector_type == "dino":
            self.detector = DINODetector(detector_config.get("model_name", "facebook/dinov2-base"))
        else:
            raise ValueError(f"Unknown detector: {detector_type}")
            
        if SAM_AVAILABLE:
            self.device = self.detector.device
            if os.path.exists(sam_checkpoint): # wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
                rospy.loginfo(f"Loading SAM from {sam_checkpoint}...")
                sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
                sam.to(device=self.device)
                self.sam_predictor = SamPredictor(sam)
                rospy.loginfo("SAM loaded!")
            else:
                rospy.logwarn(f"SAM checkpoint not found: {sam_checkpoint}")
                # Try loading without checkpoint if possible or fail?
                # Usually need checkpoint.
                self.sam_predictor = None
        else:
            self.sam_predictor = None

    def detect_and_segment(self, image_input: Union[str, np.ndarray, Image.Image], prompt: str = None, **kwargs) -> Dict:
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
            image_np = np.array(image)
        elif isinstance(image_input, np.ndarray):
            image_np = image_input
            image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        elif isinstance(image_input, Image.Image):
            image = image_input
            image_np = np.array(image)
        else:
            raise ValueError("Invalid image input type")
            
        detection = self.detector.detect(image, prompt, **kwargs)
        
        masks = []
        if self.sam_predictor and detection['found']:
            self.sam_predictor.set_image(image_np)
            for bbox in detection['bboxes']:
                # Ensure bbox is [x1, y1, x2, y2]
                mask, _, _ = self.sam_predictor.predict(box=np.array(bbox)[None, :], multimask_output=False)
                masks.append(mask[0])
                
        return {'detector': self.detector_type, 'detection': detection, 'segmentation': {'masks': masks, 'num_masks': len(masks)}}
