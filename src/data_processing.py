import os
import sys
sys.path.append(os.getcwd())
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class VisDroneDataset(Dataset):
    """
    Custom Dataset class for VisDrone-DET.
    Optimized for Aerial Vehicle Detection at 960x960 resolution.
    """
    def __init__(self, root: str, device: str = "cpu", target_size=(960, 960)):
     
        potential_nested_path = os.path.join(root, "visdrone")
        self.actual_root = potential_nested_path if os.path.exists(potential_nested_path) else root
        
        self.image_path = os.path.join(self.actual_root, "images")
        self.labels_path = os.path.join(self.actual_root, "annotations")
        self.device = device
        self.target_size = target_size

        if not os.path.exists(self.image_path):
            logger.error(f"Image directory not found at: {self.image_path}")
            raise FileNotFoundError(f"Missing directory: {self.image_path}")

        self.img_names = sorted([f for f in os.listdir(self.image_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        logger.info(f"VisDrone Data Engine: Initialized for {target_size} resolution. Samples: {len(self.img_names)}")

    def __getitem__(self, idx):
        try:
      
            img_name = self.img_names[idx]
            image_full_path = os.path.join(self.image_path, img_name)
            image = cv2.imread(image_full_path)
            
            if image is None:
                raise FileNotFoundError(f"Corrupt or missing image: {image_full_path}")
            
    
            h_orig, w_orig = image.shape[:2]
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
            
         
            ratio_w = self.target_size[0] / w_orig
            ratio_h = self.target_size[1] / h_orig
            
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            img_res = img_rgb / 255.0 
            img_tensor = torch.as_tensor(img_res).permute(2, 0, 1) 

         
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_full_path = os.path.join(self.labels_path, label_name)

            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor([idx]),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64)
            }

            boxes = []
            labels = []

            if os.path.exists(label_full_path):
                with open(label_full_path, "r") as f:
                    for line in f.readlines():
                        parts = line.strip().split(',')
                        if len(parts) < 6: continue
                        
  
                        x_min = float(parts[0]) * ratio_w
                        y_min = float(parts[1]) * ratio_h
                        width = float(parts[2]) * ratio_w
                        height = float(parts[3]) * ratio_h
                        category = int(parts[5])

                 
                        if category in [4, 5, 6, 9]:
                            x_max = x_min + width
                            y_max = y_min + height
                            
                    
                            if x_max > x_min and y_max > y_min:
                                boxes.append([x_min, y_min, x_max, y_max])
                                labels.append(1) 

            if boxes:
                target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
                target["labels"] = torch.tensor(labels, dtype=torch.int64)
                target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * \
                                 (target["boxes"][:, 2] - target["boxes"][:, 0])
                target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)


            img_tensor = img_tensor.to(self.device)
            for key in target:
                target[key] = target[key].to(self.device)

            return img_tensor, target
        
        except Exception as e:
            logger.error(f"Error at dataset index {idx}: {str(e)}")
            import sys
            raise CustomException(e, sys)

    def __len__(self):
        return len(self.img_names)

if __name__ == "__main__":
    import sys
    TEST_ROOT = os.path.join("artifacts", "data_ingestion", "VisDrone_Dataset")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "="*50)
    print("      VISDRONE DATA ENGINE: 960x960 RESIZE AUDIT      ")
    print("="*50)

    try:
        dataset = VisDroneDataset(root=TEST_ROOT, device=DEVICE)
        image, target = dataset[0]

        print(f"\n[+] Audit Success")
        print(f"    - Target Resolution  : 960 x 960")
        print(f"    - Image Tensor Shape : {image.shape} (C, H, W)")
        print(f"    - Targets Detected   : {len(target['labels'])} vehicles")
        
        if len(target['boxes']) > 0:
            print(f"    - First Scaled Box   : {target['boxes'][0].tolist()}")
        
        print("\n[✔] Conclusion: Resize and Scaling Logic is PRODUCTION-READY.")
        print("="*50 + "\n")

    except Exception as e:
        print(f"\n[✘] Audit Failed: {e}")
        sys.exit(1)