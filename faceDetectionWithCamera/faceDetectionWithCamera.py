# فایل face detection whit camera

# اتصال به دیتابیس های redis و mysql

# اتصال در لحظه و همیشگی به دوربین از بین تصویری که تحت نظر دارد به دنبال چهره یک انسان میگردد .
# هرگاه چهره ای را تشخیص داده از آن عکس میگیرد .
# عکس گرفته شده از چهره را با توجه به پارامتر هایی که برای چهره ی هر انسانی متفاوت است ، encode  میکند Encode مذکور را
# با تمامی encode های ذخیره شده از قبل در دیتا بیس redis از نظر مطابقت بررسی می کند
# حالا که به اطلاعات کاربری شخصی که از روبروی دوربین رد شده و دسترسی داریم اطلاعات آن را در جدول حضور غیاب از دیتا بیس mysql ذخیره می نماییم


import cv2
import redis
import base64
import numpy as np
from persiantools.jdatetime import JalaliDateTime
import mysql.connector
from datetime import datetime

# اتصال به Redis
redis_connection = redis.StrictRedis(host='localhost', port=6379, decode_responses=True)

# اتصال به MySQL
mysql_connection = mysql.connector.connect(
    host='localhost',
    database='face_recognition',
    user='root',
    password='1234'
)

# استفاده از کلاسی برای تشخیص چهره
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# بارگذاری مدل تشخیص چهره OpenCV برای شناسایی (LBPH یا مدل‌های پیشرفته)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainer/model.xml')  # بارگذاری مدل آموزش‌دیده

# تابع اصلاح پدینگ Base64
def correct_base64_padding(base64_string):
    padding = len(base64_string) % 4
    if padding != 0:
        base64_string += '=' * (4 - padding)
    return base64_string

# تبدیل تصویر رشته Base64 به تصویر OpenCV
def base64_to_cv2(base64_string):
    base64_string = correct_base64_padding(base64_string)  # اصلاح پدینگ قبل از دیکد کردن
    try:
        decoded_data = base64.b64decode(base64_string)  # دیکد کردن داده‌ها
        np_data = np.frombuffer(decoded_data, np.uint8)  # تبدیل به داده‌های عددی
        image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)  # تبدیل به تصویر OpenCV
        return image
    except Exception as e:
        print(f"Error in decoding base64 string: {e}")
        return None  # در صورت بروز خطا، None برمی‌گرداند

# تبدیل تصویر OpenCV به رشته Base64
def cv2_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

# فلگ برای بررسی ثبت تکراری
is_person_detected = False

# اتصال به دوربین
camera = cv2.VideoCapture(0)

while True:
    # خواندن تصویر از دوربین
    ret, frame = camera.read()

    # تشخیص چهره‌ها در تصویر
    faces = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # نمایش مستطیل دور چهره‌ها و بررسی تشخیص هر چهره با موجودیت‌های قبلی
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # مربع سبز برای چهره‌ها

        # برش تصویر چهره
        face_roi = frame[y:y + h, x:x + w]

        # تبدیل تصویر چهره به رشته Base64
        face_encoded_str = cv2_to_base64(face_roi)

        # جستجو در دیتابیس Redis برای کلید‌هایی که به عنوان کد ملی شناخته شده‌اند
        matched = False  # فلگ برای نشان دادن اینکه فرد شناسایی شده است
        for key in redis_connection.scan_iter():
            national_code = key  # Redis keys are already strings in Python 3

            # دریافت تصویر چهره از دیتابیس Redis
            face_from_redis = redis_connection.get(national_code)
            if face_from_redis:
                face_decoded = base64_to_cv2(face_from_redis)

                # تبدیل تصویر چهره به خاکستری (grayscale) قبل از پیش‌بینی
                face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

                # مقایسه چهره‌ها با استفاده از LBPH یا روش‌های پیچیده‌تر
                label, confidence = face_recognizer.predict(face_roi_gray)

                if confidence < 100 and not is_person_detected:
                    matched = True  # فرد شناسایی شده است
                    print(f"Face matched with national code: {national_code} with confidence: {confidence}")

                    # بررسی وجود رکورد‌های قبلی در جدول MatchedPersons با تاریخ فعلی
                    current_date = datetime.now().strftime('%Y-%m-%d')
                    cursor = mysql_connection.cursor()
                    cursor.execute("SELECT COUNT(*) FROM MatchedPersons WHERE nationalCode = %s AND detectionTime LIKE %s",
                                   (national_code, f"{current_date}%"))
                    existing_records_count = cursor.fetchone()[0]

                    # اگر هیچ رکوردی با تاریخ مشخص در اون روز یافت نشد، اطلاعات را در جدول ذخیره کنید
                    if existing_records_count == 0:
                        # دریافت اطلاعات فرد از دیتابیس MySQL
                        cursor.execute("SELECT firstName, lastName FROM NewPerson WHERE nationalCode = %s",
                                       (national_code,))
                        person_data = cursor.fetchone()

                        # بررسی اینکه آیا داده‌های فرد موجود هستند
                        if person_data:
                            first_name, last_name = person_data
                            # ذخیره تاریخ و زمان شمسی فعلی
                            current_jalali_time = JalaliDateTime.now().strftime('%Y-%m-%d %H:%M:%S')

                            # ذخیره اطلاعات در جدول جدید
                            cursor.execute(
                                "INSERT INTO MatchedPersons (nationalCode, firstName, lastName, detectionTime) VALUES (%s, %s, %s, %s)",
                                (national_code, first_name, last_name, current_jalali_time))
                            mysql_connection.commit()
                            is_person_detected = True
                        else:
                            print(f"No data found for national code: {national_code}")
                    else:
                        print(f"This person has already been detected today.")

        # اگر فرد شناسایی شده است، نام او را زیر تصویر نشان می‌دهیم
        if matched:
            cv2.putText(frame, f"Detected: {national_code}", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"Not Detected", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # نمایش تصویر باز شده
    cv2.imshow('Camera', frame)

    # ایجاد امکان خروج از حلقه با فشردن کلید 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# آزاد کردن دوربین و بستن پنجره‌های نمایش
camera.release()
cv2.destroyAllWindows()


    