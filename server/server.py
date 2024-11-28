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


# import base64
# import numpy as np
# import redis
# import cv2
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from persiantools.jdatetime import JalaliDateTime
# import json
# import os
#
# # تنظیمات سرور و Redis
# redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)
# app = Flask(__name__)
# CORS(app)
#
# # بررسی مسیر Haar Cascade
# face_cascade_path = "assets/face_detection/haarcascade_frontalface_default.xml"
# left_eye_cascade_path = "assets/face_detection/haarcascade_left_eye.xml"
# right_eye_cascade_path = "assets/face_detection/haarcascade_right_eye.xml"
#
# if not os.path.exists(face_cascade_path):
#     raise FileNotFoundError(f"Haar Cascade file not found at {face_cascade_path}")
#
# face_cascade = cv2.CascadeClassifier(face_cascade_path)
#
# # Load or initialize LBPH face recognizer
# model = cv2.face.LBPHFaceRecognizer_create()
# model_path = "trainer/model.xml"
#
# # بررسی اینکه پوشه ذخیره‌سازی مدل وجود دارد یا نه
# model_dir = os.path.dirname(model_path)
# if not os.path.exists(model_dir):
#     os.makedirs(model_dir)  # اگر پوشه وجود ندارد، آن را می‌سازیم
#
# # بررسی اینکه فایل مدل وجود دارد یا نه
# if os.path.exists(model_path):
#     model.read(model_path)
#
# # تبدیل تصویر Base64 به OpenCV
# def base64_to_cv2_image(base64_str):
#     img_data = base64.b64decode(base64_str.split(',')[1])
#     np_arr = np.frombuffer(img_data, np.uint8)
#     img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#     return img
#
# # پردازش تصویر و تشخیص چهره کامل
# def detect_and_validate_face(image):
#     gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)
#
#     if len(faces) == 0:
#         return None, None
#
#     for (x, y, w, h) in faces:
#         face = gray_img[y:y+h, x:x+w]
#         face = cv2.resize(face, (200, 200))
#
#         # بررسی اینکه چهره کامل باشه (چشم، بینی، دهان)
#         eyes_detected = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml').detectMultiScale(face)
#         if len(eyes_detected) < 2:
#             return None, None  # چهره کامل نیست
#
#         return face, (x, y, w, h)
#
#     return None, None
# # استخراج ویژگی‌های چهره (embedding)
# def extract_face_embedding(face):
#     # تبدیل چهره به ویژگی با استفاده از LBPH
#     label, confidence = model.predict(face)
#     return confidence  # بازگشت به میزان اعتماد به جای ویژگی‌ها
#
# # ذخیره اطلاعات در Redis
# def save_to_redis(national_code, first_name, last_name, face_image):
#     # تبدیل تصویر به Base64
#     _, buffer = cv2.imencode('.jpg', face_image)
#     base64_face = base64.b64encode(buffer).decode('utf-8')
#
#     current_time = JalaliDateTime.now().strftime('%Y-%m-%d %H:%M:%S')
#     face_data = {
#         "firstName": first_name,
#         "lastName": last_name,
#         "faceImage": base64_face,  # ذخیره تصویر به صورت Base64
#         "detectionTime": current_time
#     }
#     redis_client.set(national_code, json.dumps(face_data))
#
# # روت API برای آپلود تصویر
# @app.route('/upload', methods=['POST'])
# def upload_image():
#     data = request.json
#     image_data = data.get("image")
#     national_code = data.get("nationalCode")
#     first_name = data.get("firstName")
#     last_name = data.get("lastName")
#
#     if not image_data or not national_code or not first_name or not last_name:
#         return jsonify({"status": "error", "message": "کلیه فیلدها باید پر شوند"}), 400
#
#     try:
#         image = base64_to_cv2_image(image_data)
#         face, _ = detect_and_validate_face(image)
#
#         if face is None:
#             return jsonify({"status": "error", "message": "چهره کامل شناسایی نشد. لطفاً تصویر واضح و کامل ارائه دهید."}), 400
#
#         # استخراج ویژگی چهره
#         face_embedding = extract_face_embedding(face)
#
#         # ذخیره اطلاعات در Redis و آموزش مدل
#         save_to_redis(national_code, first_name, last_name, face_embedding)
#
#         # به‌روزرسانی مدل با صورت و برچسب‌ها
#         labels = np.array([int(national_code)], dtype=np.int32)  # تبدیل به آرایه numpy با نوع داده صحیح
#         faces = [face]
#
#         # به‌روزرسانی مدل با صورت و برچسب‌ها
#         model.update(faces, labels)
#
#         # ذخیره مدل
#         model.write(model_path)
#
#         return jsonify({"status": "success", "message": "چهره با موفقیت شناسایی و ذخیره شد"})
#
#     except Exception as e:
#         return jsonify({"status": "error", "message": f"خطا در پردازش تصویر: {str(e)}"}), 500  # اجرای سرور
#
#
# if __name__ == '__main__':
#     app.run(debug=True)
