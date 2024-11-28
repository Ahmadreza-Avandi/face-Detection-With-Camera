import base64
import numpy as np
import redis
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from persiantools.jdatetime import JalaliDateTime
import json
import os
import logging

# تنظیمات لاگ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# تنظیمات Redis و Flask
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)
app = Flask(__name__)
CORS(app)

# بارگذاری Haar Cascade
HAAR_CASCADE_PATHS = {
    "face": "assets/face_detection/haarcascade_frontalface_default.xml",
    "eye": "assets/face_detection/haarcascade_eye.xml"
}

if not all(os.path.exists(path) for path in HAAR_CASCADE_PATHS.values()):
    raise FileNotFoundError("یکی از فایل‌های Haar Cascade موجود نیست.")

face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATHS["face"])
eye_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATHS["eye"])

# تبدیل تصویر Base64 به OpenCV
def base64_to_cv2_image(base64_str):
    try:
        img_data = base64.b64decode(base64_str.split(',')[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        logging.error(f"خطا در تبدیل تصویر Base64: {e}")
        raise ValueError("تصویر معتبر نیست.")

# پردازش تصویر و تشخیص چهره کامل
def detect_and_validate_face(image):
    try:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            return None, None

        for (x, y, w, h) in faces:
            face = gray_img[y:y + h, x:x + w]
            face = cv2.resize(face, (200, 200))

            # بررسی اینکه چهره کامل باشد (چشم‌ها)
            eyes_detected = eye_cascade.detectMultiScale(face)
            if len(eyes_detected) < 2:
                logging.warning("چهره ناقص است: چشم‌ها شناسایی نشدند.")
                return None, None

            return face, (x, y, w, h)

        return None, None
    except Exception as e:
        logging.error(f"خطا در پردازش تصویر: {e}")
        raise

# ذخیره اطلاعات در Redis
def save_to_redis(national_code, first_name, last_name, face_image):
    try:
        _, buffer = cv2.imencode('.jpg', face_image)
        base64_face = base64.b64encode(buffer).decode('utf-8')

        face_data = {
            "firstName": first_name,
            "lastName": last_name,
            "faceImage": base64_face,
            "detectionTime": JalaliDateTime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        redis_client.set(national_code, json.dumps(face_data))
        logging.info(f"اطلاعات برای کد ملی {national_code} با موفقیت ذخیره شد.")
    except Exception as e:
        logging.error(f"خطا در ذخیره اطلاعات در Redis: {e}")
        raise ValueError("ذخیره‌سازی در Redis با خطا مواجه شد.")

# اعتبارسنجی ورودی‌ها
def validate_inputs(data):
    required_fields = ["image", "nationalCode", "firstName", "lastName"]
    for field in required_fields:
        if not data.get(field):
            raise ValueError(f"فیلد {field} الزامی است.")
    return True

# روت آپلود تصویر
@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        data = request.json

        # اعتبارسنجی ورودی‌ها
        validate_inputs(data)

        # تبدیل تصویر Base64 به OpenCV
        image = base64_to_cv2_image(data["image"])

        # تشخیص و تایید چهره
        face, _ = detect_and_validate_face(image)
        if face is None:
            return jsonify({"status": "error", "message": "چهره شناسایی نشد یا چهره ناقص است"}), 400

        # ذخیره چهره در Redis
        save_to_redis(data["nationalCode"], data["firstName"], data["lastName"], face)

        return jsonify({"status": "success", "message": "اطلاعات با موفقیت ذخیره شد."})

    except ValueError as ve:
        logging.error(f"خطای ورودی: {ve}")
        return jsonify({"status": "error", "message": str(ve)}), 400
    except Exception as e:
        logging.error(f"خطا در آپلود تصویر: {e}")
        return jsonify({"status": "error", "message": "خطا در پردازش تصویر"}), 500

if __name__ == "__main__":
    app.run(debug=True)

#-----------------------------------------------------------------------


# 