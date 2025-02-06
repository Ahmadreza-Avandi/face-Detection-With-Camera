import cv2
import numpy as np
import mysql.connector
from datetime import datetime
from persiantools.jdatetime import JalaliDateTime
import schedule
import time
import logging

# تنظیمات لاگینگ
logging.basicConfig(
    level=logging.INFO,  # برای مشاهده اطلاعات بیشتر، این سطح را به DEBUG تغییر دهید
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --------------------- کلاس مدیریت دوربین‌ها ---------------------
class CameraManager:
    def __init__(self):
        self.cameras = []
        self.grid_size = (2, 2)  # (تعداد ردیف‌ها، تعداد ستون‌ها)
        self.active_cam = -1  # حالت تمام صفحه: -1 یعنی حالت گرید
        self.window_name = "Face Recognition System"  # نام پنجره نمایش
        self.last_click = 0
        self.click_delay = 500  # میلی‌ثانیه

        # بارگذاری مدل تشخیص چهره
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_recognizer.read('trainer/model.xml')

        # تنظیم اتصال به دیتابیس
        try:
            self.db = mysql.connector.connect(
                host='localhost',
                database='face_recognition',
                user='root',
                password='1234'
            )
            logger.info("اتصال به دیتابیس برقرار شد.")
        except mysql.connector.Error as err:
            logger.error(f"خطا در اتصال به دیتابیس: {err}")
            self.db = None

        # در این دیکشنری، برای هر کاربر زمان و مکان آخرین حضور ذخیره می‌شود.
        self.last_checkin = {}

    def add_camera(self, name, source, location):
        """
        اضافه کردن دوربین به لیست مدیریت
        اگر منبع دوربین عددی (مثلاً 0) نباشد، آن را به عنوان دوربین خارجی (مثلاً الوهاست) در نظر می‌گیریم
        """
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            # تعیین نوع دوربین: True یعنی دوربین خارجی (مداربسته) که نیاز به شبیه‌سازی فاصله کانونی دارد
            is_external = False if isinstance(source, int) and source == 0 else True
            self.cameras.append({
                'cap': cap,
                'name': name,
                'location': location,
                'frame': None,
                'is_external': is_external  # مشخص‌کننده اینکه آیا دوربین خارجی است یا نه
            })
            logger.info(f"دوربین '{name}' در '{location}' فعال شد!")
        else:
            logger.error(f"خطا در اتصال به دوربین '{name}'")
            cap.release()

    def adjust_focal_distance(self, frame, zoom_factor=1.5):
        """
        شبیه‌سازی تنظیم فاصله کانونی با استفاده از تکنیک زوم دیجیتال (کراس کردن قسمت میانی فریم)
        پارامتر zoom_factor میزان زوم را تعیین می‌کند؛ مقدار بزرگتر به معنی زوم بیشتر (کراس کوچکتر) است.
        """
        h, w = frame.shape[:2]
        new_w = int(w / zoom_factor)
        new_h = int(h / zoom_factor)
        x1 = (w - new_w) // 2
        y1 = (h - new_h) // 2
        cropped = frame[y1:y1+new_h, x1:x1+new_w]
        adjusted = cv2.resize(cropped, (640, 480))
        return adjusted

    def process_faces(self, frame, location):
        """
        پردازش چهره‌ها در فریم دریافتی:
         - تبدیل فریم به سیاه و سفید
         - اعمال هیستوگرام سازگاری برای افزایش کنتراست
         - تشخیص چهره‌ها و رسم مستطیل دور آن‌ها
         - شناسایی چهره و ثبت حضور در دیتابیس در صورت شناسایی صحیح
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_roi = gray[y:y + h, x:x + w]
            label, confidence = self.face_recognizer.predict(face_roi)
            if confidence < 100:
                self.log_attendance(str(label), location)

        return cv2.resize(frame, (640, 480))

    def log_attendance(self, national_code, location):
        """
        ثبت حضور در جدول attendance و به‌روزرسانی آخرین حضور در latest_attendance.
        منطق جدید:
          - اگر کاربر برای اولین بار دیده شده باشد، رکورد جدید در هر دو جدول ثبت می‌شود.
          - اگر کاربر قبلاً دیده شده باشد، در صورتی که از حضور قبلی بیش از 2 ساعت گذشته باشد یا
            لوکیشن (دوربین) تغییر کرده باشد، رکورد جدید در جدول attendance ثبت می‌شود.
          - در هر صورت، جدول latest_attendance به روز می‌شود.
        """
        if self.db is None:
            logger.error("دیتابیس متصل نیست. حضور ثبت نخواهد شد.")
            return

        now = datetime.now()
        jalali_time = JalaliDateTime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor = self.db.cursor()

        should_insert_attendance = False
        if national_code in self.last_checkin:
            last_time, last_location = self.last_checkin[national_code]
            time_diff = (now - last_time).total_seconds()
            if time_diff >= 7200 or last_location != location:
                should_insert_attendance = True
        else:
            should_insert_attendance = True

        if should_insert_attendance:
            try:
                insert_attendance = """
                    INSERT INTO attendance (national_code, checkin_time, location)
                    VALUES (%s, %s, %s)
                """
                cursor.execute(insert_attendance, (national_code, jalali_time, location))
                self.db.commit()
            except Exception as e:
                logger.error(f"خطای ثبت حضور در جدول attendance: {e}")

        try:
            select_user = "SELECT first_name, last_name FROM NewPerson WHERE national_code = %s"
            cursor.execute(select_user, (national_code,))
            user = cursor.fetchone()
            if user:
                first_name, last_name = user
            else:
                first_name, last_name = "نامشخص", "نامشخص"
        except Exception as e:
            logger.error(f"خطا در دریافت اطلاعات کاربر: {e}")
            first_name, last_name = "نامشخص", "نامشخص"

        try:
            insert_latest = """
                INSERT INTO latest_attendance (national_code, first_name, last_name, last_seen, location)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE 
                    first_name = VALUES(first_name),
                    last_name = VALUES(last_name),
                    last_seen = VALUES(last_seen),
                    location = VALUES(location)
            """
            cursor.execute(insert_latest, (national_code, first_name, last_name, jalali_time, location))
            self.db.commit()
            self.last_checkin[national_code] = (now, location)
            logger.info(f"حضور کاربر {national_code} ({first_name} {last_name}) در {location} ثبت شد")
        except Exception as e:
            logger.error(f"خطای به‌روز رسانی latest_attendance: {e}")

    def update_frames(self):
        """
        به‌روز کردن فریم‌های هر دوربین:
         - خواندن فریم از هر دوربین
         - در صورت موفقیت در خواندن، اعمال تنظیم فاصله کانونی برای دوربین‌های خارجی (شبیه‌سازی زوم)
         - پردازش فریم برای تشخیص چهره
         - در صورت عدم دریافت فریم، استفاده از یک فریم سیاه به عنوان جایگزین
        """
        for cam in self.cameras:
            ret, frame = cam['cap'].read()
            if ret:
                if cam.get('is_external', False):
                    frame = self.adjust_focal_distance(frame, zoom_factor=1.5)
                cam['frame'] = self.process_faces(frame, cam['location'])
            else:
                cam['frame'] = np.zeros((480, 640, 3), dtype=np.uint8)

    def toggle_fullscreen(self, x, y):
        """
        تغییر حالت نمایش:
         - از حالت گرید به حالت تمام صفحه و بالعکس با دو کلیک موس
        """
        current_time = cv2.getTickCount()
        if (current_time - self.last_click) * 1000 / cv2.getTickFrequency() < self.click_delay:
            return

        if self.active_cam == -1:
            col = x // (640 + 10)
            row = y // (480 + 10)
            idx = row * self.grid_size[1] + col
            if idx < len(self.cameras):
                self.active_cam = idx
        else:
            self.active_cam = -1

        self.last_click = current_time

    def show_interface(self):
        """
        نمایش رابط کاربری:
         - در حالت تمام صفحه، فقط فریم یک دوربین نمایش داده می‌شود.
         - در حالت گرید، فریم‌های دوربین‌ها به صورت شبکه‌ای نمایش داده می‌شوند.
         - در صورت عدم اتصال هیچ دوربینی، یک فریم سیاه نمایش داده می‌شود.
        """
        if self.active_cam != -1:
            frame = self.cameras[self.active_cam]['frame']
            cv2.imshow(self.window_name, frame)
        else:
            if not self.cameras:
                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.imshow(self.window_name, black_frame)
                return

            grid_frames = []
            for i in range(0, len(self.cameras), self.grid_size[1]):
                row_frames = [cam['frame'] if cam['frame'] is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                              for cam in self.cameras[i:i + self.grid_size[1]]]
                while len(row_frames) < self.grid_size[1]:
                    row_frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
                grid_frames.append(np.hstack(row_frames))
            final_grid = np.vstack(grid_frames[:self.grid_size[0]])
            cv2.imshow(self.window_name, final_grid)


# --------------------- تنظیمات سیستم ---------------------
def main():
    manager = CameraManager()

    # اضافه کردن دوربین‌ها:
    manager.add_camera("دوربین لپتاپ", 0, "دوربین لپتاپ")
    manager.add_camera("نماز خونه", "rtsp://admin:shn123456789@192.168.1.101:554/cam/realmonitor?channel=1", "نمازخونه")

    schedule.every(2).hours.do(manager.last_checkin.clear)

    def mouse_handler(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            manager.toggle_fullscreen(x, y)

    cv2.namedWindow(manager.window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(manager.window_name, np.zeros((480, 640, 3), dtype=np.uint8))

    try:
        cv2.setMouseCallback(manager.window_name, mouse_handler)
    except cv2.error as e:
        logger.error(f"خطا در تنظیم رویدادهای ماوس: {e}")

    if not manager.cameras:
        logger.error("هیچ دوربینی متصل نشد. برنامه در حالت شبیه‌سازی ادامه می‌یابد.")

    try:
        while True:
            manager.update_frames()
            manager.show_interface()
            schedule.run_pending()

            if cv2.waitKey(1) == 27:  # خروج با کلید ESC
                break
    finally:
        for cam in manager.cameras:
            cam['cap'].release()
        cv2.destroyAllWindows()
        if manager.db:
            manager.db.close()


if __name__ == '__main__':
    main()

