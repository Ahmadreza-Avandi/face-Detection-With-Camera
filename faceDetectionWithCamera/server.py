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
import mysql.connector

os.makedirs("trainer", exist_ok=True)
# تنظیمات لاگ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --------------------- تنظیمات اتصال ---------------------
# تنظیمات Redis
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

# تنظیمات MySQL
mysql_connection = mysql.connector.connect(
    host='localhost',
    database='face_recognition',
    user='root',
    password='1234'
)

# تنظیمات Flask و CORS
app = Flask(__name__)
CORS(app)

# --------------------- تنظیمات Haar Cascade ---------------------
HAAR_CASCADE_PATHS = {
    "face": "assets/face_detection/haarcascade_frontalface_default.xml",
    "eye": "assets/face_detection/haarcascade_eye.xml"
}

if not all(os.path.exists(path) for path in HAAR_CASCADE_PATHS.values()):
    raise FileNotFoundError("یکی از فایل‌های Haar Cascade موجود نیست.")

face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATHS["face"])
eye_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATHS["eye"])

# --------------------- توابع کمکی ---------------------
def base64_to_cv2_image(base64_str):
    """تبدیل رشته Base64 به تصویر OpenCV"""
    try:
        # اگر رشته شامل پیشوند data URI است، آن را جدا می‌کنیم
        img_data = base64.b64decode(base64_str.split(',')[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        logging.error(f"خطا در تبدیل تصویر Base64: {e}")
        raise ValueError("تصویر معتبر نیست.")

def detect_and_validate_face(image):
    """تشخیص چهره و اعتبارسنجی آن (وجود حداقل ۲ چشم)"""
    try:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            return None, None

        for (x, y, w, h) in faces:
            face = gray_img[y:y + h, x:x + w]
            face = cv2.resize(face, (200, 200))

            # بررسی وجود چشم‌ها در ناحیه چهره
            eyes_detected = eye_cascade.detectMultiScale(face)
            if len(eyes_detected) < 2:
                logging.warning("چهره ناقص است: چشم‌ها شناسایی نشدند.")
                return None, None

            return face, (x, y, w, h)

        return None, None
    except Exception as e:
        logging.error(f"خطا در پردازش تصویر: {e}")
        raise

def train_model():
    """
    آموزش مدل با داده‌های موجود در Redis
    """
    faces = []
    labels = []
    labels_to_name = {}

    # دریافت داده‌ها از Redis
    for key in redis_client.scan_iter():
        data = json.loads(redis_client.get(key))
        face_base64 = data['faceImage']
        np_arr = np.frombuffer(base64.b64decode(face_base64), np.uint8)
        face_image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)  # تبدیل به خاکستری

        if face_image is not None:
            # تغییر اندازه تصویر به 100x100
            resized_face = cv2.resize(face_image, (100, 100))
            faces.append(resized_face)
            labels.append(int(key))  # فرض: شماره ملی به عنوان لیبل
            labels_to_name[int(key)] = {
                "full_name": f"{data['firstName']} {data['lastName']}",
                "student_id": str(key)  # تبدیل به رشته
            }

    if faces and labels:
        # آموزش مدل
        model = cv2.face.LBPHFaceRecognizer_create()
        model.train(np.array(faces), np.array(labels))
        model.write("trainer/model.xml")  # ذخیره مدل

        # ذخیره لیبل‌ها در فایل JSON
        with open('labels_to_name.json', 'w', encoding='utf-8') as json_file:
            json.dump(labels_to_name, json_file, ensure_ascii=False, indent=4)

        logging.info("مدل با موفقیت آموزش داده شد و ذخیره گردید.")
    else:
        logging.warning("هیچ داده‌ای برای آموزش یافت نشد.")

def save_to_redis(national_code, first_name, last_name, face_image):
    """ذخیره اطلاعات کاربر در Redis"""
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
        logging.info(f"اطلاعات برای کد ملی {national_code} در Redis با موفقیت ذخیره شد.")
    except Exception as e:
        logging.error(f"خطا در ذخیره اطلاعات در Redis: {e}")
        raise ValueError("ذخیره‌سازی در Redis با خطا مواجه شد.")

def save_to_mysql(national_code, first_name, last_name, face_image):
    """ذخیره اطلاعات کاربر در جدول NewPerson در MySQL"""
    cursor = None
    try:
        cursor = mysql_connection.cursor()

        # بررسی وجود کاربر در جدول NewPerson
        cursor.execute("SELECT COUNT(*) FROM NewPerson WHERE national_code = %s", (national_code,))
        count = cursor.fetchone()[0]

        if count == 0:
            # تبدیل تصویر به باینری جهت ذخیره در ستون longblob
            _, buffer = cv2.imencode('.jpg', face_image)
            image_bytes = buffer.tobytes()

            cursor.execute(
                "INSERT INTO NewPerson (national_code, first_name, last_name, image) VALUES (%s, %s, %s, %s)",
                (national_code, first_name, last_name, image_bytes)
            )
            mysql_connection.commit()
            logging.info(f"کاربر {national_code} در جدول NewPerson ثبت شد.")
        else:
            logging.info(f"کاربر {national_code} از قبل در جدول NewPerson موجود است.")
    except Exception as e:
        logging.error(f"خطا در ذخیره اطلاعات در MySQL: {e}")
        raise
    finally:
        if cursor:
            cursor.close()

def validate_inputs(data):
    """اعتبارسنجی ورودی‌های دریافتی از کلاینت"""
    required_fields = ["image", "nationalCode", "firstName", "lastName"]
    for field in required_fields:
        if not data.get(field):
            raise ValueError(f"فیلد {field} الزامی است.")
    return True

# --------------------- روت آپلود تصویر ---------------------
@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        data = request.json

        # اعتبارسنجی ورودی‌ها
        validate_inputs(data)

        # تبدیل تصویر Base64 به تصویر OpenCV
        image = base64_to_cv2_image(data["image"])

        # تشخیص و تایید چهره
        face, _ = detect_and_validate_face(image)
        if face is None:
            return jsonify({"status": "error", "message": "چهره شناسایی نشد یا چهره ناقص است"}), 400

        # ذخیره در Redis
        save_to_redis(data["nationalCode"], data["firstName"], data["lastName"], face)

        # ثبت اطلاعات کاربر در MySQL (جدول NewPerson)
        save_to_mysql(data["nationalCode"], data["firstName"], data["lastName"], face)

        # آموزش مدل با داده‌های جدید
        train_model()

        return jsonify({"status": "success", "message": "اطلاعات با موفقیت ذخیره شد و مدل به‌روزرسانی گردید."})

    except ValueError as ve:
        logging.error(f"خطای ورودی: {ve}")
        return jsonify({"status": "error", "message": str(ve)}), 400
    except Exception as e:
        logging.error(f"خطا در آپلود تصویر: {e}")
        return jsonify({"status": "error", "message": "خطا در پردازش تصویر"}), 500

if __name__ == "__main__":
    app.run(debug=True)
