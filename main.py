import sys
import os
sys.path.append(os.getcwd())
import io
import os
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw, ImageFont
from src.model_architecture import FasterRCNNModel 


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join("artifacts", "models", "fasterrcnn.pth")


app = FastAPI(title="VisDrone Aerial Surveillance API")


model_factory = FasterRCNNModel(num_classes=2, device=DEVICE)
model = model_factory.model

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"✅ Loaded Sovereign Weights from {MODEL_PATH}")
else:
    print("⚠️ WARNING: Trained weights not found. Using pretrained COCO weights (Accuracy may be lower).")

model.eval()


def predict_and_draw(image: Image.Image):

    img_rgb = image.convert("RGB")
    img_array = np.array(img_rgb) / 255.0
    img_tensor = torch.as_tensor(img_array).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)


    with torch.no_grad():
        predictions = model(img_tensor)

    prediction = predictions[0]
    boxes = prediction['boxes'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()


    draw = ImageDraw.Draw(img_rgb)
    
    detected_count = 0
    for box, score in zip(boxes, scores):
    
        if score > 0.5:
            detected_count += 1
            x_min, y_min, x_max, y_max = box
            

            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
            
  
            draw.text((x_min, y_min - 10), f"Vehicle: {score:.2f}", fill="red")
    
    print(f"[*] Analysis Complete: Detected {detected_count} targets.")
    return img_rgb



@app.get("/")
def read_root():
    return {
        "Project": "VisDrone Aerial Surveillance",
        "Architecture": "Anchor-Optimized Faster R-CNN (16px)",
        "Status": "System Active",
        "Framework": "FastAPI / PyTorch"
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))


    output_image = predict_and_draw(image)


    img_byte_arr = io.BytesIO()
    output_image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
 
    uvicorn.run(app, host="0.0.0.0", port=8080)