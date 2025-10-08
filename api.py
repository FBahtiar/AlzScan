# app.py
from flask import Flask, request, jsonify
import torch
import base64
import io
from PIL import Image
import traceback
import os
import numpy as np

# Import model dari file terpisah
from model import ModifiedIntegratedModel

# Coba impor nibabel untuk NIfTI
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    print("⚠️ nibabel not installed. NIfTI support disabled.")
    NIBABEL_AVAILABLE = False

# ==================== NIfTI PROCESSING ====================

def normalize_intensity(vol, low=1, high=99):
    low_val, high_val = np.percentile(vol, [low, high])
    vol = np.clip(vol, low_val, high_val)
    vol = (vol - low_val) / (high_val - low_val)
    return (vol * 255).astype(np.uint8)

def find_best_slice(vol, axis=2, candidates=5):
    center = vol.shape[axis] // 2
    start = max(0, center - candidates // 2)
    end = min(vol.shape[axis], center + candidates // 2 + 1)
    
    best_slice, best_std = None, 0
    for i in range(start, end):
        sl = vol[i] if axis == 0 else vol[:, i] if axis == 1 else vol[:, :, i]
        std = np.std(sl)
        if std > best_std:
            best_std, best_slice = std, sl
    return best_slice if best_slice is not None else vol[:, :, vol.shape[2] // 2]

def process_nifti_to_image(nifti_data):
    nii = nib.load(io.BytesIO(nifti_data))
    vol = nii.get_fdata()
    
    if len(vol.shape) == 3:
        sl = find_best_slice(vol, axis=2)
    elif len(vol.shape) == 4:
        sl = find_best_slice(vol[..., 0], axis=2)
    else:
        raise ValueError(f"Unsupported shape: {vol.shape}")
    
    img = normalize_intensity(sl)
    rgb = np.stack([img] * 3, axis=-1) if img.ndim == 2 else img
    return Image.fromarray(rgb)

# ==================== UTILS ====================

def detect_file_type(data):
    if len(data) >= 4:
        if data[:4] in (b'n+1\x00', b'ni1\x00'):
            return 'nifti'
        if data[:3] == b'\x1f\x8b\x08':  # gzip
            return 'nifti'
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return 'png'
    if data[:3] == b'\xff\xd8\xff':
        return 'jpeg'
    return 'unknown'

def create_rotation_invariant_prediction(model, image, transform, device, class_names):
    rotations = [0, 90, 180, 270]
    all_probs, preds = [], []

    for angle in rotations:
        img_rot = image.rotate(angle, expand=False)
        tensor = transform(img_rot).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
            preds.append(class_names[pred_idx])
            all_probs.append(probs)

    avg_probs = np.mean(all_probs, axis=0)
    final_idx = np.argmax(avg_probs)
    final_pred = class_names[final_idx]
    confidence = avg_probs[final_idx]
    consistency = preds.count(final_pred) / len(preds)

    return {
        "prediction": final_pred,
        "confidence": float(confidence),
        "avg_probabilities": avg_probs.tolist(),
        "consistency": consistency,
        "individual_predictions": [
            {"rotation": r, "prediction": p, "confidence": float(conf)}
            for r, p, conf in zip(rotations, preds, [p[np.argmax(p)] for p in all_probs])
        ]
    }

# ==================== FLASK APP ====================

app = Flask(__name__)

# Konfigurasi
class_names = ['ad', 'mci', 'nor']
num_classes = len(class_names)

best_config = {
    'hidden_size': 1792,
    'num_of_attention_heads': 8,
    'dropout_rate': 0.3
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")

# Load model
MODEL_PATH = "/home/nathasyasiregar/opsi_alzheimer/env_alzheimer/best_model.pth"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = ModifiedIntegratedModel(best_config, num_classes).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
model.eval()
print("✅ Model loaded.")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== ROUTES ====================

@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'Missing image'}), 400

        file_data = base64.b64decode(data['image'])
        file_type = detect_file_type(file_data)
        print(f"🔍 File type: {file_type}")

        if file_type == 'nifti':
            if not NIBABEL_AVAILABLE:
                return jsonify({'error': 'Install nibabel for NIfTI support'}), 400
            image = process_nifti_to_image(file_data)
            info = {'input_type': 'NIfTI', 'converted_to': 'RGB_image'}
        else:
            image = Image.open(io.BytesIO(file_data)).convert('RGB')
            info = {'input_type': 'regular_image', 'format': file_type}

        result = create_rotation_invariant_prediction(model, image, transform, device, class_names)
        avg_probs = result['avg_probabilities']

        return jsonify({
            'success': True,
            'label': result['prediction'],
            'confidence': round(result['confidence'], 4),
            'probabilities': {class_names[i]: round(avg_probs[i], 4) for i in range(num_classes)},
            'rotation_ensemble': {
                'consistency_score': round(result['consistency'], 3),
                'individual_results': result['individual_predictions']
            },
            'processing_info': info,
            'model_used': 'EfficientNetV2-L + MHSA',
            'device': str(device)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'device': str(device),
        'classes': class_names,
        'nifti_support': NIBABEL_AVAILABLE
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=False)# app.py
from flask import Flask, request, jsonify
import torch
import base64
import io
from PIL import Image
import traceback
import os
import numpy as np

# Import model dari file terpisah
from model import ModifiedIntegratedModel

# Coba impor nibabel untuk NIfTI
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    print("⚠️ nibabel not installed. NIfTI support disabled.")
    NIBABEL_AVAILABLE = False

# ==================== NIfTI PROCESSING ====================

def normalize_intensity(vol, low=1, high=99):
    low_val, high_val = np.percentile(vol, [low, high])
    vol = np.clip(vol, low_val, high_val)
    vol = (vol - low_val) / (high_val - low_val)
    return (vol * 255).astype(np.uint8)

def find_best_slice(vol, axis=2, candidates=5):
    center = vol.shape[axis] // 2
    start = max(0, center - candidates // 2)
    end = min(vol.shape[axis], center + candidates // 2 + 1)
    
    best_slice, best_std = None, 0
    for i in range(start, end):
        sl = vol[i] if axis == 0 else vol[:, i] if axis == 1 else vol[:, :, i]
        std = np.std(sl)
        if std > best_std:
            best_std, best_slice = std, sl
    return best_slice if best_slice is not None else vol[:, :, vol.shape[2] // 2]

def process_nifti_to_image(nifti_data):
    nii = nib.load(io.BytesIO(nifti_data))
    vol = nii.get_fdata()
    
    if len(vol.shape) == 3:
        sl = find_best_slice(vol, axis=2)
    elif len(vol.shape) == 4:
        sl = find_best_slice(vol[..., 0], axis=2)
    else:
        raise ValueError(f"Unsupported shape: {vol.shape}")
    
    img = normalize_intensity(sl)
    rgb = np.stack([img] * 3, axis=-1) if img.ndim == 2 else img
    return Image.fromarray(rgb)

# ==================== UTILS ====================

def detect_file_type(data):
    if len(data) >= 4:
        if data[:4] in (b'n+1\x00', b'ni1\x00'):
            return 'nifti'
        if data[:3] == b'\x1f\x8b\x08':  # gzip
            return 'nifti'
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return 'png'
    if data[:3] == b'\xff\xd8\xff':
        return 'jpeg'
    return 'unknown'

def create_rotation_invariant_prediction(model, image, transform, device, class_names):
    rotations = [0, 90, 180, 270]
    all_probs, preds = [], []

    for angle in rotations:
        img_rot = image.rotate(angle, expand=False)
        tensor = transform(img_rot).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
            preds.append(class_names[pred_idx])
            all_probs.append(probs)

    avg_probs = np.mean(all_probs, axis=0)
    final_idx = np.argmax(avg_probs)
    final_pred = class_names[final_idx]
    confidence = avg_probs[final_idx]
    consistency = preds.count(final_pred) / len(preds)

    return {
        "prediction": final_pred,
        "confidence": float(confidence),
        "avg_probabilities": avg_probs.tolist(),
        "consistency": consistency,
        "individual_predictions": [
            {"rotation": r, "prediction": p, "confidence": float(conf)}
            for r, p, conf in zip(rotations, preds, [p[np.argmax(p)] for p in all_probs])
        ]
    }

# ==================== FLASK APP ====================

app = Flask(__name__)

# Konfigurasi
class_names = ['ad', 'mci', 'nor']
num_classes = len(class_names)

best_config = {
    'hidden_size': 1792,
    'num_of_attention_heads': 16,
    'dropout_rate': 0.3
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")

# Load model
MODEL_PATH = "/home/nathasyasiregar/opsi_alzheimer/env_alzheimer/best_model.pth"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = ModifiedIntegratedModel(best_config, num_classes).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
model.eval()
print("✅ Model loaded.")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== ROUTES ====================

@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'Missing image'}), 400

        file_data = base64.b64decode(data['image'])
        file_type = detect_file_type(file_data)
        print(f"🔍 File type: {file_type}")

        if file_type == 'nifti':
            if not NIBABEL_AVAILABLE:
                return jsonify({'error': 'Install nibabel for NIfTI support'}), 400
            image = process_nifti_to_image(file_data)
            info = {'input_type': 'NIfTI', 'converted_to': 'RGB_image'}
        else:
            image = Image.open(io.BytesIO(file_data)).convert('RGB')
            info = {'input_type': 'regular_image', 'format': file_type}

        result = create_rotation_invariant_prediction(model, image, transform, device, class_names)
        avg_probs = result['avg_probabilities']

        return jsonify({
            'success': True,
            'label': result['prediction'],
            'confidence': round(result['confidence'], 4),
            'probabilities': {class_names[i]: round(avg_probs[i], 4) for i in range(num_classes)},
            'rotation_ensemble': {
                'consistency_score': round(result['consistency'], 3),
                'individual_results': result['individual_predictions']
            },
            'processing_info': info,
            'model_used': 'EfficientNetV2-L + MHSA',
            'device': str(device)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'device': str(device),
        'classes': class_names,
        'nifti_support': NIBABEL_AVAILABLE
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=False)
