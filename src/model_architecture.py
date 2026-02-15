import sys
import os
sys.path.append(os.getcwd())

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from src.logger import get_logger
from src.custom_exception import CustomException 

logger = get_logger(__name__)

class FasterRCNNModel:
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device

        self.model = self.create_model().to(self.device)
        logger.info("VisDrone Architecture Initialized: Anchor Scales optimized for Tiny Targets.")
    
    def create_model(self):
        try:
  
            anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )

    
            model = fasterrcnn_resnet50_fpn(
                pretrained=True,
                rpn_anchor_generator=rpn_anchor_generator
            )

           
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, 
                self.num_classes
            )
            
            return model

        except Exception as e:
            logger.error(f"Architectural Error: Failed to generate Anchor-Optimized R-CNN {e}")
            import sys
            raise CustomException(e, sys)

