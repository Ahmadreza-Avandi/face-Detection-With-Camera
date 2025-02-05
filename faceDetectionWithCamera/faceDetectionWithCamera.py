import cv2
import numpy as np
import mysql.connector
from datetime import datetime
from persiantools.jdatetime import JalaliDateTime
import schedule
import time


# --------------------- کلاس مدیریت دوربین‌ها ---------------------
class CameraManager:
    def __init__(self):
        self.cameras = []
        self.grid_size = (2, 2)  # (تعداد ردیف‌ها، تعداد ستون‌ها)
        self.active_cam = -1  # حالت تمام صفحه: -1 یعنی حالت گرید
        self.window_name = "Face Recognition System"  # تغییر نام به انگلیسی برای اطمینان از سازگاری
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
            print("✅ اتصال به دیتابیس برقرار شد.")
        except mysql.connector.Error as err:
            print(f"❌ خطا در اتصال به دیتابیس: {err}")
            self.db = None
        self.last_checkin = {}

    def add_camera(self, name, source, location):
        """اضافه کردن دوربین به لیست مدیریت"""
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            self.cameras.append({
                'cap': cap,
                'name': name,
                'location': location,
                'frame': None
            })
            print(f"✅ دوربین '{name}' در '{location}' فعال شد!")
        else:
            print(f"❌ خطا در اتصال به دوربین '{name}'")
            # آزادسازی شی cap در صورت عدم موفقیت
            cap.release()

    def process_faces(self, frame, location):
        """پردازش چهره‌ها در فریم دریافتی"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # افزایش دقت تشخیص با اعمال هیستوگرام سازگاری و بلور گوسی
        gray = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # رسم مستطیل دور چهره تشخیص داده‌شده
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_roi = gray[y:y + h, x:x + w]

            # شناسایی چهره با استفاده از LBPH
            label, confidence = self.face_recognizer.predict(face_roi)
            if confidence < 100:
                # ثبت حضور در دیتابیس
                self.log_attendance(str(label), location)

        return cv2.resize(frame, (640, 480))

    def log_attendance(self, national_code, location):
        """
        ثبت حضور در جدول attendance و به‌روزرسانی آخرین حضور در latest_attendance.
        از جلوگیری از ثبت حضور مکرر در بازه 2 ساعته نیز پشتیبانی می‌کند.
        """
        if self.db is None:
            print("❌ دیتابیس متصل نیست. حضور ثبت نخواهد شد.")
            return

        now = datetime.now()
        if national_code in self.last_checkin:
            if (now - self.last_checkin[national_code]).total_seconds() < 7200:
                return  # اگر کمتر از 2 ساعت از ثبت حضور قبلی گذشته باشد، ثبت نمی‌کند

        try:
            cursor = self.db.cursor()

            # زمان به‌روز شده به فرمت جلالی
            jalali_time = JalaliDateTime.now().strftime('%Y-%m-%d %H:%M:%S')

            # ثبت رکورد در جدول attendance
            insert_attendance = """
                INSERT INTO attendance (national_code, checkin_time, location)
                VALUES (%s, %s, %s)
            """
            cursor.execute(insert_attendance, (national_code, jalali_time, location))
            self.db.commit()

            # دریافت اطلاعات کاربری (نام و نام خانوادگی) از جدول NewPerson
            select_user = "SELECT first_name, last_name FROM NewPerson WHERE national_code = %s"
            cursor.execute(select_user, (national_code,))
            user = cursor.fetchone()

            if user:
                first_name, last_name = user
            else:
                first_name, last_name = "نامشخص", "نامشخص"

            # ثبت یا به‌روزرسانی آخرین حضور در جدول latest_attendance
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

            self.last_checkin[national_code] = now
            print(f"✅ حضور کاربر {national_code} ({first_name} {last_name}) در {location} ثبت شد")
        except Exception as e:
            print(f"❌ خطای دیتابیس: {e}")

    def update_frames(self):
        """به‌روز کردن فریم‌های هر دوربین"""
        for cam in self.cameras:
            ret, frame = cam['cap'].read()
            if ret:
                cam['frame'] = self.process_faces(frame, cam['location'])
            else:
                cam['frame'] = np.zeros((480, 640, 3), dtype=np.uint8)

    def toggle_fullscreen(self, x, y):
        """تغییر حالت نمایش: از نمایش گرید به حالت تمام صفحه"""
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
        """نمایش رابط کاربری: حالت گرید دوربین‌ها با فضای خالی برای دوربین‌های غیرفعال"""
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

    # اضافه کردن دوربین‌ها
    manager.add_camera("دوربین لپتاپ", 0, "دوربین لپتاپ")
    manager.add_camera("نماز خانه", "rtsp://admin:shn123456789@192.168.1.101:554/cam/realmonitor?channel=1",
                       "نماز خانه")

    # زمان‌بندی پاکسازی last_checkin هر 2 ساعت
    schedule.every(2).hours.do(manager.last_checkin.clear)

    # تعریف تابع مدیریت رویدادهای ماوس
    def mouse_handler(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            manager.toggle_fullscreen(x, y)

    # ایجاد پنجره نمایش به همراه یک تصویر اولیه
    cv2.namedWindow(manager.window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(manager.window_name, np.zeros((480, 640, 3), dtype=np.uint8))

    try:
        cv2.setMouseCallback(manager.window_name, mouse_handler)
    except cv2.error as e:
        print(f"❌ خطا در تنظیم رویدادهای ماوس: {e}")

    if not manager.cameras:
        print("❌ هیچ دوربینی متصل نشد. برنامه در حالت شبیه‌سازی ادامه می‌یابد.")

    # حلقه اصلی برنامه
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
