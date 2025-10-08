from flask import Flask, request, jsonify, render_template
import requests
import json
import base64
import traceback
import nibabel as nib
import numpy as np
from PIL import Image
import io
import os
import tempfile

app = Flask(__name__)


API_TOKEN = "403bfb7198314769a528a05a082f7bd8"

API_URL = "https://node380-ai-hub.ub.ac.id/user/nathasyasiregar/proxy/5002/classify"

# === ROUTES ===
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'mode': 'remote_with_nifti_support',
        'api_url': API_URL,
        'nifti_rules': '192→118, 240→140, 256→160'
    })

@app.route('/get_nifti_preview', methods=['POST'])
def get_nifti_preview():
    """Endpoint baru untuk mendapatkan preview slice NIFTI"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400

        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return jsonify({'success': False, 'error': 'Empty filename'}), 400

        file_type = detect_file_type(uploaded_file.filename)
        
        if file_type != 'NIFTI':
            return jsonify({'success': False, 'error': 'File is not NIFTI format'}), 400

        try:
            # Process NIFTI file
            slice_data, slice_info = extract_nifti_slice_data(uploaded_file)
            
            # Convert slice to base64 image
            slice_image = Image.fromarray(slice_data.astype(np.uint8))
            slice_image = slice_image.resize((224, 224))
            
            # Convert to base64
            buffered = io.BytesIO()
            slice_image.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            return jsonify({
                'success': True,
                'slice_image': img_base64,
                'slice_index': slice_info['selected_slice'],
                'total_slices': slice_info['total_slices'],
                'dimensions': f"{slice_info['width']}x{slice_info['height']}x{slice_info['total_slices']}"
            })
            
        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed to process NIFTI: {str(e)}'}), 400
            
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'Tidak ada file diunggah.'}), 400

        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return jsonify({'success': False, 'error': 'Nama file kosong.'}), 400

        file_type = detect_file_type(uploaded_file.filename)
        processed_image_bytes = None

        # Konversi NIFTI ke gambar jika diperlukan
        if file_type == 'NIFTI':
            try:
                img = convert_nifti_to_image(uploaded_file)
                buffered = io.BytesIO()
                # Simpan sebagai JPEG untuk kompatibilitas
                img.save(buffered, format="JPEG") 
                processed_image_bytes = buffered.getvalue()
                file_type = 'Image'
                print("File NIFTI berhasil dikonversi ke JPEG.")
            except Exception as e:
                return jsonify({'success': False, 'error': f'Gagal memproses file NIFTI: {str(e)}'}), 400

        # Jika file adalah gambar langsung
        elif file_type == 'Image':
             uploaded_file.seek(0) # Reset pointer file
             processed_image_bytes = uploaded_file.read()
        else:
             return jsonify({'success': False, 'error': f'Tipe file tidak didukung: {uploaded_file.filename}'}), 400

 
        if processed_image_bytes:
            encoded_image = base64.b64encode(processed_image_bytes).decode('utf-8')
            payload = {
                'image': encoded_image
            }

            headers = {
                "Authorization": f"Bearer {API_TOKEN}",
                "Content-Type": "application/json" # Sangat penting
            }

            print(f" Mengirim gambar ke server AI: {API_URL}")
            response = requests.post(API_URL, json=payload, headers=headers, timeout=120) # Timeout diperpanjang
            response.raise_for_status() # Akan melempar exception untuk status error HTTP
            result = response.json()

            print(" Hasil dari server AI:", result)

            return jsonify({
                'success': True,
                'prediction': result.get('prediction', result.get('label', 'unknown')),
                'label': result.get('label', result.get('prediction', 'unknown')),
                'confidence': float(result.get('confidence', result.get('probability', 0.0))), # Pastikan float
                'confidence_value': float(result.get('confidence', result.get('probability', 0.0))),
                'probabilities': result.get('probabilities', {}),
                'file_type': file_type,
                'method': 'EfficientNet V2 + MHSA (Remote)',
                'preprocessing': 'Slice sesuai rules & Resize 224x224',
                'processing_info': result.get('processing_info', {}),
                'image_size_processed': '224x224'
            })
        else:
            return jsonify({'success': False, 'error': 'Gagal memproses gambar untuk dikirim.'}), 500

    except requests.exceptions.Timeout:
        error_msg = 'Timeout: Server AI terlalu lama merespons.'
        print(f"❌ {error_msg}")
        return jsonify({'success': False, 'error': error_msg}), 504 # Gateway Timeout
    except requests.exceptions.ConnectionError:
        error_msg = 'Gagal koneksi ke server AI. Pastikan server aktif dan URL benar.'
        print(f"❌ {error_msg}")
        return jsonify({'success': False, 'error': error_msg}), 502 # Bad Gateway
    except requests.exceptions.HTTPError as err:
        error_msg = f'HTTP Error dari server AI: {err}'
        print(f"❌ {error_msg}")
        try:
            error_detail = err.response.json()
            error_msg += f" Detail: {error_detail}"
        except:
            pass
        return jsonify({'success': False, 'error': error_msg}), err.response.status_code
    except Exception as e:
        traceback.print_exc()
        error_msg = f'Terjadi kesalahan di server lokal: {str(e)}'
        print(f"❌ {error_msg}")
        return jsonify({'success': False, 'error': error_msg}), 500

def detect_file_type(filename):
    if not filename:
        return 'Unknown'
    filename_lower = filename.lower()
    if filename_lower.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        return 'Image'
    elif filename_lower.endswith(('.nii', '.nii.gz')):
        return 'NIFTI'
    return 'Unknown'

def get_specific_slice(volume, axis=2):
    num_slices = volume.shape[axis]
    
    if num_slices == 192:
        slice_idx = 117
    elif num_slices == 240:
        slice_idx = 140
    elif num_slices == 256:
        slice_idx = 160
    else:
        slice_idx = num_slices // 2
  
    slice_idx = min(slice_idx, num_slices - 1)
    
    print(f"Volume slices: {num_slices}, Selected slice: {slice_idx}")
    
    if axis == 0:
        return volume[slice_idx, :, :], slice_idx
    elif axis == 1:
        return volume[:, slice_idx, :], slice_idx
    else:  # axis == 2 (axial - most common)
        return volume[:, :, slice_idx], slice_idx

def normalize_intensity(volume, percentile_low=2, percentile_high=98):
    try:
        volume_float = volume.astype(np.float64)
        volume_float = np.nan_to_num(volume_float, nan=0.0, posinf=0.0, neginf=0.0)
        non_zero_mask = volume_float > 0
        if np.sum(non_zero_mask) > 0:
            non_zero_values = volume_float[non_zero_mask]
            low_val = np.percentile(non_zero_values, percentile_low)
            high_val = np.percentile(non_zero_values, percentile_high)
        else:
            low_val = np.min(volume_float)
            high_val = np.max(volume_float)
      
        if high_val == low_val:
            return np.full_like(volume_float, 128, dtype=np.uint8)
        
        volume_clipped = np.clip(volume_float, low_val, high_val)
        volume_normalized = (volume_clipped - low_val) / (high_val - low_val)
        
        volume_scaled = (volume_normalized * 255).astype(np.uint8)
        
        return volume_scaled
        
    except Exception as e:
        print(f"⚠️ Normalization error: {str(e)}, using fallback")
        v_min, v_max = np.min(volume), np.max(volume)
        if v_max == v_min:
            return np.full_like(volume, 128, dtype=np.uint8)
        return ((volume - v_min) / (v_max - v_min) * 255).astype(np.uint8)

def extract_nifti_slice_data(file_storage):
    import gzip
    temp_file_path = None
    
    try:
        file_storage.seek(0)
        file_data = file_storage.read()
        
        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as temp_file:
            temp_file_path = temp_file.name
d
            if file_data[:3] == b'\x1f\x8b\x08':
                # Data is gzipped, decompress first
                try:
                    decompressed_data = gzip.decompress(file_data)
                    temp_file.write(decompressed_data)
                    print("🗜️ File NIFTI gzipped berhasil di-decompress untuk preview")
                except gzip.BadGzipFile:
                    # If gzip fails, try writing raw data
                    temp_file.write(file_data)
                    print("⚠️ Gagal decompress gzip untuk preview, menggunakan raw data")
            else:
                temp_file.write(file_data)
            
            temp_file.flush()

        print(f"File NIFTI disimpan sementara untuk preview: {temp_file_path}")

        nii_img = nib.load(temp_file_path)
        volume = nii_img.get_fdata()
        
        print(f"📊 NIfTI shape untuk preview: {volume.shape}")

        if len(volume.shape) == 3:
            slice_2d, selected_idx = get_specific_slice(volume, axis=2)  # axial
        elif len(volume.shape) == 4:
            volume_3d = volume[:, :, :, 0]
            slice_2d, selected_idx = get_specific_slice(volume_3d, axis=2)
        else:
            raise ValueError(f"Unsupported volume shape: {volume.shape}")

        if slice_2d is None or slice_2d.size == 0:
            raise ValueError("Invalid slice extracted from NIfTI volume")

        if np.max(slice_2d) == np.min(slice_2d):
            print("⚠️ Warning: Slice might be empty for preview, using fallback")

            middle_idx = volume.shape[2] // 2
            slice_2d = volume[:, :, middle_idx]
            selected_idx = middle_idx

        print(f"Selected slice index untuk preview: {selected_idx}")

        normalized_slice = normalize_intensity(slice_2d)

        if len(normalized_slice.shape) == 2:

            rgb_slice = np.stack([normalized_slice] * 3, axis=-1)
        else:
            rgb_slice = normalized_slice

        slice_info = {
            'selected_slice': selected_idx,
            'total_slices': volume.shape[2] if len(volume.shape) >= 3 else 1,
            'width': volume.shape[0],
            'height': volume.shape[1]
        }
        
        print(f"Successfully extracted NIFTI slice for preview: {rgb_slice.shape}")
        return rgb_slice, slice_info
        
    except Exception as e:
        print(f"Gagal extract slice NIFTI untuk preview: {e}")
        traceback.print_exc()
        raise
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"File sementara preview berhasil dihapus: {temp_file_path}")

def convert_nifti_to_image(file_storage):
    import gzip
    temp_file_path = None
    
    try:
        file_storage.seek(0)
        file_data = file_storage.read()
        
        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as temp_file:
            temp_file_path = temp_file.name
            
            if file_data[:3] == b'\x1f\x8b\x08':
                # Data is gzipped, decompress first
                try:
                    decompressed_data = gzip.decompress(file_data)
                    temp_file.write(decompressed_data)
                    print("🗜️ File NIFTI gzipped berhasil di-decompress")
                except gzip.BadGzipFile:
                    temp_file.write(file_data)
                    print("⚠️ Gagal decompress gzip, menggunakan raw data")
            else:
                temp_file.write(file_data)
            
            temp_file.flush()

        print(f"File NIFTI disimpan sementara di: {temp_file_path}")

        nii_img = nib.load(temp_file_path)
        volume = nii_img.get_fdata()
        
        print(f"NIfTI shape: {volume.shape}")

        if len(volume.shape) == 3:
            slice_2d, selected_idx = get_specific_slice(volume, axis=2)  # axial
        elif len(volume.shape) == 4:
            volume_3d = volume[:, :, :, 0]
            slice_2d, selected_idx = get_specific_slice(volume_3d, axis=2)
        else:
            raise ValueError(f"Unsupported volume shape: {volume.shape}")

        if slice_2d is None or slice_2d.size == 0:
            raise ValueError("Invalid slice extracted from NIfTI volume")

        if np.max(slice_2d) == np.min(slice_2d):
            print("⚠️ Warning: Slice might be empty, using fallback")
            middle_idx = volume.shape[2] // 2
            slice_2d = volume[:, :, middle_idx]
            selected_idx = middle_idx

        print(f"Selected slice index: {selected_idx}")
        normalized_slice = normalize_intensity(slice_2d)
        if len(normalized_slice.shape) == 2:
            rgb_slice = np.stack([normalized_slice] * 3, axis=-1)
        else:
            rgb_slice = normalized_slice
        
        img = Image.fromarray(rgb_slice.astype(np.uint8))
        img = img.resize((224, 224))  # Resize ke 224x224 sesuai model
        
        print(f"Successfully processed NIfTI to image: {img.size}")
        return img
        
    except Exception as e:
        print(f"❌ Gagal memproses file NIFTI: {e}")
        traceback.print_exc()
        raise
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"🗑️ File sementara berhasil dihapus: {temp_file_path}")

if __name__ == '__main__':
    print("Flask server berjalan di http://localhost:5002")
    print(f"Terhubung ke server AI UB: {API_URL}")
    print("NIfTI slice rules: 192→118, 240→140, 256→160")
    print("NIfTI preview support: Enabled")
    app.run(debug=True, port=5002, host='0.0.0.0')
