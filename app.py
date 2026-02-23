"""
Smart Car Parking System - Complete Application
Advanced OCR with extreme conditions support
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import base64
import os
from datetime import datetime
import easyocr
import re
from config import get_db_connection
import time

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__, 
            static_folder=os.path.join(basedir, 'static'),
            static_url_path='/static',
            template_folder=os.path.join(basedir, 'templates'))

UPLOAD_FOLDER = os.path.join(basedir, 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

print("⏳ Loading EasyOCR model...")
reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='./model')
print("✅ EasyOCR loaded successfully!")


def decode_base64_image(base64_string):
    """Convert Base64 to OpenCV image"""
    try:
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]
        img_bytes = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"❌ Decode error: {e}")
        return None


def fix_dark_image(image):
    """Brighten dark images"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    brightened = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return brightened


def fix_angled_image(image):
    """Correct perspective distortion"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
            rect = order_points(pts)
            (tl, tr, br, bl) = rect
            
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            maxWidth = max(int(widthA), int(widthB))
            
            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            maxHeight = max(int(heightA), int(heightB))
            
            dst = np.array([[0, 0], [maxWidth - 1, 0], 
                           [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], 
                          dtype="float32")
            
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
            return warped
    
    return image


def order_points(pts):
    """Order corner points"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def enhance_image(image):
    """Multi-level enhancement"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    filtered = cv2.bilateralFilter(enhanced, 5, 50, 50)
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    return thresh


def detect_plate_region(image):
    """Detect license plate area"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blur, 30, 200)
    
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
        
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            
            if 2.0 <= aspect_ratio <= 5.5 and w > 100 and h > 30:
                pad = 10
                x = max(0, x - pad)
                y = max(0, y - pad)
                w = min(image.shape[1] - x, w + 2*pad)
                h = min(image.shape[0] - y, h + 2*pad)
                return image[y:y+h, x:x+w]
    
    # Fallback to center region
    h, w = image.shape[:2]
    return image[int(h*0.35):int(h*0.75), int(w*0.15):int(w*0.85)]


def perform_ocr(plate_image, enhanced):
    """Multi-attempt OCR"""
    all_results = []
    
    # Attempt 1: Original
    try:
        results = reader.readtext(plate_image, detail=1, batch_size=1)
        for bbox, text, conf in results:
            all_results.append((text, conf, 'original'))
    except:
        pass
    
    # Attempt 2: Enhanced
    try:
        results = reader.readtext(enhanced, detail=1, batch_size=1)
        for bbox, text, conf in results:
            all_results.append((text, conf, 'enhanced'))
    except:
        pass
    
    # Attempt 3: Scaled
    try:
        scaled = cv2.resize(plate_image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        results = reader.readtext(scaled, detail=1, batch_size=1)
        for bbox, text, conf in results:
            all_results.append((text, conf, 'scaled'))
    except:
        pass
    
    return all_results


def clean_and_validate(text):
    """Clean and validate plate text"""
    text = text.upper().strip()
    text = re.sub(r'[^A-Z0-9]', '', text)
    
    # Skip invalid
    if text in ['IND', 'INDIA', 'BHARAT'] or len(text) < 6:
        return None, 0.0
    
    # Apply corrections
    result = ""
    for i, char in enumerate(text):
        if i < 2:  # State code
            result += char
        elif i < 4:  # District
            result += '0' if char == 'O' else ('1' if char == 'I' else char)
        elif i < 7:  # Series
            result += 'O' if char == '0' else ('I' if char == '1' else char)
        else:  # Numbers
            result += '0' if char == 'O' else ('1' if char == 'I' else char)
    
    # Validate pattern
    pattern = r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{3,4}$'
    if re.match(pattern, result):
        return result, 1.15
    elif len(result) >= 8:
        return result, 1.0
    
    return None, 0.0


def select_best_result(all_results):
    """Select highest confidence valid result"""
    valid = []
    
    for text, conf, method in all_results:
        cleaned, boost = clean_and_validate(text)
        if cleaned:
            valid.append((cleaned, conf * boost, method))
    
    if not valid:
        return None, 0.0, None
    
    valid.sort(key=lambda x: x[1], reverse=True)
    return valid[0]


def format_plate(text):
    """Format: TN21AW2122 → TN21 AW 2122"""
    if len(text) >= 8:
        return f"{text[0:4]} {text[4:6]} {text[6:]}"
    return text


def save_image(image, prefix="capture"):
    """Save image to uploads folder"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{prefix}_{timestamp}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return filename


def process_parking_record_with_flag(plate, confidence, image_path):
    """Handle database entry/exit with flag"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute(
            "SELECT id FROM parking_records WHERE plate=%s AND flag='1' AND out_time IS NULL LIMIT 1",
            (plate,)
        )
        existing = cursor.fetchone()
        
        if existing:
            cursor.execute(
                "UPDATE parking_records SET out_time=NOW(3), flag='0', ocr_confidence=%s WHERE id=%s",
                (confidence, existing['id'])
            )
            conn.commit()
            action, record_id = "exit", existing['id']
        else:
            cursor.execute(
                "INSERT INTO parking_records (plate, flag, ocr_confidence, image_path, in_time) VALUES (%s,'1',%s,%s,NOW(3))",
                (plate, confidence, image_path)
            )
            conn.commit()
            action, record_id = "entry", cursor.lastrowid
        
        cursor.close()
        conn.close()
        return action, record_id
    except Exception as e:
        print(f"❌ DB Error: {e}")
        return "error", None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/capture', methods=['POST'])
def capture():
    """Process captured image"""
    start = time.time()
    
    try:
        data = request.get_json()
        base64_image = data.get('image')
        
        if not base64_image:
            return jsonify({"success": False, "error": "No image"}), 400
        
        print("\n" + "="*60)
        print("🚀 PROCESSING")
        print("="*60)
        
        # Decode
        image = decode_base64_image(base64_image)
        if image is None:
            return jsonify({"success": False, "error": "Decode failed"}), 400
        
        # Save original
        original_file = save_image(image, "original")
        
        # Enhance images
        dark_fixed = fix_dark_image(image)
        angle_fixed = fix_angled_image(dark_fixed)
        
        # Detect plate
        plate_roi = detect_plate_region(angle_fixed)
        enhanced = enhance_image(plate_roi)
        
        # Save ROI
        roi_file = save_image(plate_roi, "plate")
        
        # Perform OCR
        all_results = perform_ocr(plate_roi, enhanced)
        
        # Select best
        best_text, confidence, method = select_best_result(all_results)
        
        if not best_text:
            print(f"❌ FAILED | Time: {time.time()-start:.2f}s")
            print("="*60)
            return jsonify({
                "success": False,
                "error": "No valid plate detected. Ensure good lighting and clear view."
            }), 400
        
        # Format
        formatted = format_plate(best_text)
        
        # Database
        action, record_id = process_parking_record_with_flag(formatted, confidence, original_file)
        
        if action == "error":
            return jsonify({"success": False, "error": "Database error"}), 500
        
        total = time.time() - start
        
        print(f"✅ SUCCESS")
        print(f"   Plate: {formatted}")
        print(f"   Confidence: {confidence:.2%}")
        print(f"   Action: {action.upper()}")
        print(f"   Time: {total:.2f}s")
        print("="*60)
        
        return jsonify({
            "success": True,
            "plate": formatted,
            "confidence": round(confidence, 2),
            "action": action,
            "record_id": record_id,
            "image_path": original_file,
            "roi_path": roi_file,
            "processing_time": round(total, 2)
        })
    
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚗 SMART PARKING SYSTEM")
    print("="*60)
    print("✅ Advanced OCR with extreme conditions support")
    print("✅ Works in dark, angled, blurry images")
    print("✅ Entry/Exit tracking with flag system")
    print("="*60)
    print("\n🌐 Open: http://127.0.0.1:5000")
    print("⏹️  Stop: CTRL+C\n")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)