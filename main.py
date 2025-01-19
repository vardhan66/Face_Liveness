import os
import datetime
import pickle
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import face_recognition
import util
from test import test


class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520+350+100")
        self.main_window.title("Face Liveness System")

        self.logged_in_user = None  # Track currently logged-in user

        # Buttons
        self.login_button_main_window = util.get_button(self.main_window, 'Login', 'green', self.login)
        self.login_button_main_window.place(x=750, y=150)

        self.logout_button_main_window = util.get_button(self.main_window, 'Logout', 'red', self.logout)
        self.logout_button_main_window.place(x=750, y=250)

        self.register_new_user_button_main_window = util.get_button(
            self.main_window, 'Register New User', 'gray', self.register_new_user, fg='black')
        self.register_new_user_button_main_window.place(x=750, y=350)

        # Webcam feed
        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)
        self.add_webcam(self.webcam_label)

        # Database setup
        self.setup_database()

    def setup_database(self):
        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)
        self.log_path = './log.txt'

    def add_webcam(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)

        self._label = label
        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()
        if ret:
            self.most_recent_capture_arr = frame
            img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
            self.most_recent_capture_pil = Image.fromarray(img_)
            imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
            self._label.imgtk = imgtk
            self._label.configure(image=imgtk)
        self._label.after(20, self.process_webcam)

    def login(self):
        label = test(
            image=self.most_recent_capture_arr,
            model_dir='./resources/anti_spoof_models',
            device_id=0
        )
        if label == 1:
            name = self.recognize_user(self.most_recent_capture_arr)
            if name in ['unknown_person', 'no_persons_found']:
                util.msg_box('Oops...', 'Unknown user. Please register a new user or try again.')
            else:
                if self.logged_in_user:
                    util.msg_box('Error', f'User "{self.logged_in_user}" is already logged in.')
                else:
                    self.logged_in_user = name  # Set the logged-in user
                    util.msg_box('Welcome back!', f'Welcome, {name}.')
                    with open(self.log_path, 'a') as f:
                        f.write(f'{name},{datetime.datetime.now()},in\n')
        else:
            util.msg_box('Access Denied', 'Spoof Detetced!')

    def logout(self):
        if not self.logged_in_user:  # Check if a user is logged in
            util.msg_box('Error', 'No user is currently logged in.')
            return

        label = test(
            image=self.most_recent_capture_arr,
            model_dir='./resources/anti_spoof_models',
            device_id=0
        )
        if label == 1:
            name = self.recognize_user(self.most_recent_capture_arr)
            if name == self.logged_in_user:  # Ensure the same user is logging out
                util.msg_box('Goodbye!', f'See you again, {name}.')
                with open(self.log_path, 'a') as f:
                    f.write(f'{name},{datetime.datetime.now()},out\n')
                self.logged_in_user = None  # Reset the logged-in user
            else:
                util.msg_box('Oops...', 'Authentication failed. Please try again.')
        else:
            util.msg_box('Access Denied', 'Liveness detection failed!')

    def register_new_user(self):
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("1200x520+370+120")

        self.accept_button_register_new_user_window = util.get_button(
            self.register_new_user_window, 'Accept', 'green', self.accept_register_new_user)
        self.accept_button_register_new_user_window.place(x=750, y=300)

        self.try_again_button_register_new_user_window = util.get_button(
            self.register_new_user_window, 'Try Again', 'red', self.try_again_register_new_user)
        self.try_again_button_register_new_user_window.place(x=750, y=400)

        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)
        self.add_img_to_label(self.capture_label)

        self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=750, y=150)

        self.text_label_register_new_user = util.get_text_label(
            self.register_new_user_window, 'Please input username:')
        self.text_label_register_new_user.place(x=750, y=70)

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def recognize_user(self, image):
        known_faces = []
        known_names = []
        for file in os.listdir(self.db_dir):
            if file.endswith('.pickle'):
                name = os.path.splitext(file)[0]
                with open(os.path.join(self.db_dir, file), 'rb') as f:
                    known_faces.append(pickle.load(f))
                    known_names.append(name)
        face_encodings = face_recognition.face_encodings(image)
        if len(face_encodings) > 0:
            face = face_encodings[0]
            distances = face_recognition.face_distance(known_faces, face)
            min_distance_index = distances.argmin() if len(distances) > 0 else None
            if min_distance_index is not None and distances[min_distance_index] < 0.6:  # Use threshold for confidence
                return known_names[min_distance_index]
        return 'unknown_person'

    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get(1.0, "end-1c").strip()
        if not name:
            util.msg_box('Error!', 'Username cannot be empty!')
            return
        if os.path.exists(os.path.join(self.db_dir, f'{name}.pickle')):
            util.msg_box('Error!', f'Username "{name}" already exists!')
            return
        face_encodings = face_recognition.face_encodings(self.register_new_user_capture)
        if len(face_encodings) == 0:
            util.msg_box('Error!', 'No face detected. Please try again.')
            return
        with open(os.path.join(self.db_dir, f'{name}.pickle'), 'wb') as file:
            pickle.dump(face_encodings[0], file)
        util.msg_box('Success!', f'User "{name}" registered successfully!')
        self.register_new_user_window.destroy()

    def start(self):
        self.main_window.mainloop()


if __name__ == "__main__":
    app = App()
    app.start()
