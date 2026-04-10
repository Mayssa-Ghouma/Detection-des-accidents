"""
AccidentAI — Serveur Flask (YOLOv8) - VERSION CORRIGÉE (Windows Safe)
====================================
Lancement :
    pip install flask flask-cors ultralytics pillow opencv-python
    python app.py
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io, base64, os, time, json
import cv2  
import tempfile

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'saved_models', 'best_yolo.pt')
IMG_SIZE   = 224
CLASSES    = ['Accident', 'Non Accident'] 

# ── App ────────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=BASE_DIR, static_url_path='')
CORS(app)

# ── Chargement du modèle YOLO ──────────────────────────────────────────────────
print('⏳ Chargement du modèle YOLOv8...')
model = None
try:
    model = YOLO(MODEL_PATH)
    dummy = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color=(128, 128, 128))
    model.predict(dummy, verbose=False, imgsz=IMG_SIZE)
    print(f'✅ Modèle YOLOv8 chargé : {MODEL_PATH}')
except Exception as e:
    print(f'❌ Erreur chargement modèle : {e}')

# ── Utilitaire de prédiction ───────────────────────────────────────────────────
def predict_from_pil(img: Image.Image):
    img_rgb = img.convert('RGB')
    t0 = time.time()
    res = model.predict(img_rgb, verbose=False, imgsz=IMG_SIZE)[0]
    elapsed = round(time.time() - t0, 3)

    accident_idx = 0
    for idx, name in res.names.items():
        if name.lower() == 'accident':
            accident_idx = idx
            break

    accident_proba = float(res.probs.data[accident_idx])
    normal_proba   = 1.0 - accident_proba
    pred_name      = res.names[int(res.probs.top1)]

    return accident_proba, normal_proba, elapsed, pred_name

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/status')
def status():
    return jsonify({
        'model_loaded' : model is not None,
        'model_name'   : 'YOLOv8s Classification'
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modèle non chargé'}), 503

    try:
        img = None
        
        # ── Cas : Fichier Multipart (Images et Vidéos) ───────
        if 'file' in request.files:
            file = request.files['file']
            filename = file.filename.lower()
            
            # --- TRAITEMENT VIDÉO (CORRECTIF WINDOWS) ---
            if filename.endswith(('.mp4', '.avi', '.mov')):
                # Utilisation de mkstemp pour un contrôle total sous Windows
                fd, temp_path = tempfile.mkstemp(suffix='.mp4')
                try:
                    # Étape 1: Écriture du fichier
                    with os.fdopen(fd, 'wb') as tmp:
                        tmp.write(file.read())
                    
                    # Étape 2: Lecture de la frame avec OpenCV
                    cap = cv2.VideoCapture(temp_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
                    ret, frame = cap.read()
                    
                    # Étape 3: FERMETURE IMMÉDIATE DU FLUX
                    cap.release() 
                    
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(frame_rgb)
                finally:
                    # Étape 4: Suppression sécurisée
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except:
                            pass # Le fichier sera nettoyé par le système plus tard
            
            # --- TRAITEMENT IMAGE ---
            else:
                img = Image.open(file.stream)

        # ── Cas : JSON Base64 ────────────────────────────────
        elif request.is_json:
            data = request.get_json()
            img_b64 = data.get('image', '')
            if ',' in img_b64:
                img_b64 = img_b64.split(',')[1]
            img_bytes = base64.b64decode(img_b64)
            img = Image.open(io.BytesIO(img_bytes))

        if img is None:
            return jsonify({'error': 'Fichier invalide'}), 400

        # Analyse
        acc_p, norm_p, elapsed, pred = predict_from_pil(img)

        return jsonify({
            'accident_proba' : round(acc_p, 4),
            'normal_proba'   : round(norm_p, 4),
            'is_accident'    : acc_p > 0.5,
            'confidence'     : round(max(acc_p, norm_p), 4),
            'elapsed_sec'    : elapsed,
            'predicted_class': pred
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)