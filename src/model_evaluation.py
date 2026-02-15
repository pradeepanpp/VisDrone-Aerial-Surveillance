import os
import sys

sys.path.append(os.getcwd())

import torch
import json
import time
from tqdm import tqdm 
from torch.utils.data import DataLoader, random_split
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from src.model_architecture import FasterRCNNModel
from src.data_processing import VisDroneDataset
from src.logger import get_logger
from src.custom_exception import CustomException 

logger = get_logger(__name__)

class ModelEvaluation:
    def __init__(self, model_path, dataset_path, device):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.device = device
        

        self.metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')

    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def evaluate(self):
        try:
            logger.info("Starting Scientific Evaluation Phase...")

   
            model_factory = FasterRCNNModel(num_classes=2, device=self.device)
            model = model_factory.model
            

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model weights not found at {self.model_path}")
            
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            logger.info("Sovereign Weights loaded successfully.")

   
            full_dataset = VisDroneDataset(self.dataset_path, self.device, target_size=(960, 960))
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            
            _, val_dataset = random_split(
                full_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=self.collate_fn)


            logger.info(f"Profiling inference on {len(val_dataset)} samples...")
            
            all_latencies = []
            
            with torch.no_grad():
                for images, targets in tqdm(val_loader, desc="Evaluating"):
                    images = list(img.to(self.device) for img in images)
                    
      
                    start_time = time.time()
                    outputs = model(images)
                    end_time = time.time()
       
                    
                    batch_latency = (end_time - start_time) / len(images)
                    all_latencies.append(batch_latency)
                    
          
                    res = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]
                    gt = [{k: v.to("cpu") for k, v in t.items()} for t in targets]
                    
                    self.metric.update(res, gt)

     
            results = self.metric.compute()
            avg_latency_ms = (sum(all_latencies) / len(all_latencies)) * 1000

   
            metrics_report = {
                "mAP_Global": round(float(results['map']), 4),
                "mAP_50_IoU": round(float(results['map_50']), 4),
                "mAP_Small_Objects": round(float(results['map_small']), 4),
                "avg_inference_latency_ms": round(avg_latency_ms, 2),
                "input_resolution": "960x960",
                "optimization_strategy": "16px_Anchors"
            }

     
            output_file = "artifacts/metrics.json"
            os.makedirs("artifacts", exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(metrics_report, f, indent=4)

            logger.info(f"Evaluation Complete. Verified mAP_small: {metrics_report['mAP_Small_Objects']}")
            
            print("\n" + "="*45)
            print(f"      SOVEREIGN PERFORMANCE REPORT      ")
            print("="*45)
            print(f"mAP (Small Targets): {metrics_report['mAP_Small_Objects']}  (PhD KPI)")
            print(f"Avg Latency:        {metrics_report['avg_inference_latency_ms']} ms")
            print("="*45 + "\n")

        except Exception as e:
            import sys
            raise CustomException(e, sys)

if __name__ == "__main__":
    MODEL_PATH = "artifacts/models/fasterrcnn.pth"
    DATA_DIR = "artifacts/data_ingestion/VisDrone_Dataset"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    evaluator = ModelEvaluation(MODEL_PATH, DATA_DIR, DEVICE)
    evaluator.evaluate()