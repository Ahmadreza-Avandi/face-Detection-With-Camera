import redis
import cv2
import numpy as np
import json
import base64
import os

# اتصال به Redis
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

# مسیر مدل و لیبل‌ها
model_path = "trainer/model.xml"
labels_to_name_path = 'labels_to_name.json'

# ایجاد پوشه برای مدل در صورت نیاز
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# بارگذاری یا ایجاد مدل
model = cv2.face.LBPHFaceRecognizer_create()
if os.path.exists(model_path):
    model.read(model_path)

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
        model.train(np.array(faces), np.array(labels))
        model.write(model_path)

        # ذخیره لیبل‌ها در فایل JSON
        with open(labels_to_name_path, 'w', encoding='utf-8') as json_file:
            json.dump(labels_to_name, json_file, ensure_ascii=False, indent=4)

        print("مدل با موفقیت آموزش داده شد و ذخیره گردید.")
    else:
        print("هیچ داده‌ای برای آموزش یافت نشد.")

if __name__ == "__main__":
    train_model()
