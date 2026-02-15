import sys
import os
sys.path.append(os.getcwd())

import torch
import time
from tqdm import tqdm # PhD Visibility Tool
from torch.utils.data import DataLoader, random_split
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from src.model_architecture import FasterRCNNModel
from src.logger import get_logger
from src.custom_exception import CustomException
from src.data_processing import VisDroneDataset

logger = get_logger(__name__)
MODEL_SAVE_PATH = "artifacts/models/"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

class ModelTraining:
    def __init__(self, num_classes, learning_rate, epochs, dataset_path, device):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dataset_path = dataset_path
        self.device = device

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"tensorboard_logs/{time.strftime('%Y%m%d-%H%M%S')}")

        try:
            model_obj = FasterRCNNModel(self.num_classes, self.device)
            self.model = model_obj.model
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            logger.info(f"Initialized Model on {self.device} with optimized anchors.")
        except Exception as e:
            import sys
            raise CustomException(e, sys)
    
    def collate_fn(self, batch):
        return tuple(zip(*batch))
    
    def prepare_data(self):
        dataset = VisDroneDataset(self.dataset_path, self.device)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # CHANGE: Batch Size reduced to 2 for laptop stability
        self.train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=self.collate_fn)
        self.val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=self.collate_fn)

        logger.info(f"Loaded {train_size} training and {val_size} validation images.")
        return self.train_loader, self.val_loader
    
    def train(self):
        try:
            train_loader, val_loader = self.prepare_data()

            for epoch in range(self.epochs):
                self.model.train()
                epoch_loss = 0
                
                # CHANGE: Added TQDM Progress Bar
                progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")

                for i, (images, targets) in enumerate(progress_bar):
                    images = list(image.to(self.device) for image in images)
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                    self.optimizer.zero_grad()
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                    losses.backward()
                    self.optimizer.step()

                    current_loss = losses.item()
                    epoch_loss += current_loss
                    
                    # Update Progress Bar with current loss
                    progress_bar.set_postfix(loss=f"{current_loss:.4f}")
                    self.writer.add_scalar("Loss/Train_Batch", current_loss, epoch * len(train_loader) + i)

                avg_train_loss = epoch_loss / len(train_loader)
                logger.info(f"Epoch {epoch+1} Finished | Avg Loss: {avg_train_loss:.4f}")
                
                # Save weights every epoch as a safety checkpoint
                torch.save(self.model.state_dict(), os.path.join(MODEL_SAVE_PATH, "fasterrcnn.pth"))

            self.writer.close()
            logger.info("Training complete. Sovereign weights saved.")
        except Exception as e:
            import sys
            raise CustomException(e, sys)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_DIR = "artifacts/data_ingestion/VisDrone_Dataset"

    training = ModelTraining(
        num_classes=2,
        learning_rate=0.0001,
        epochs=50,
        dataset_path=DATA_DIR,
        device=device
    )
    training.train()