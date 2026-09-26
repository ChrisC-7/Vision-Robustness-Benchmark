from PIL import Image
from torchvision import transforms
import torch
from torch import nn
from pathlib import Path
import json
from .models import load_model
_LABELs = {
    0:'Airplane',
    1:'Automobile',
    2: 'Bird',
    3: 'Cat',
    4: 'Deer',
    5: 'Dog',
    6: 'Frog',
    7: 'Horse',
    8: 'Ship',
    9: 'Truck'
}

def predict_img(model: nn.Module, img_path):
    img_tensor = preprocess_image(img_path)
    return predict(model, img_tensor)

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")

    transform_fn = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor()
        ]
    )
    img_tensor:torch.Tensor = transform_fn(img)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

def predict(model: nn.Module, img:torch.Tensor):
    model.eval()
    with torch.inference_mode():
        device = next(model.parameters()).device
        img = img.to(device)
        pred = model(img)
        pred_idx = pred.argmax(dim=1).item()
        prob = torch.softmax(pred, dim=1)
        pred_label = _LABELs[pred_idx]
        confidence = prob[0, pred_idx].item()

    return {
        "predicted_index": pred_idx,
        "predicted_class": pred_label,
        "confidence": confidence,
    }
    

def load_run_model(run_dir, expected_model_name, device):
    config_path = Path(run_dir)/'config.json'
    model_path = Path(run_dir)/'best.pt'
    if not config_path.exists():
            raise FileNotFoundError('Config doesn\'t exist')

    with open(config_path, 'r', encoding='UTF-8') as f:
        config_data = json.load(f)

    model_name = config_data['model_name']
    if model_name != expected_model_name:
        raise ValueError(f"Expected model:'{expected_model_name}', found '{model_name}'")
    
    if not model_path.exists():
        raise FileNotFoundError('Model doesn\'t exist')

    model = load_model(model_name, model_path, device)
    model = model.to(device)
    model.eval()
    return model

def load_model_registry(simple_run_dir, stronger_run_dir, device='cpu'):
    simple_model = load_run_model(simple_run_dir, 'simple', device)
    stronger_model = load_run_model(stronger_run_dir, 'stronger', device)
    return {
        "simple": simple_model,
        "stronger": stronger_model
    }