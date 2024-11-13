import cv2
import os
from tkinter import *
from tkinter import ttk, messagebox
from datetime import datetime
from PIL import Image, ImageTk


class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Capture Application")
        self.root.geometry("1000x600")

        self.selected_camera = None
        self.cap = None
        self.session_folder = None  # Folder for current session
        self.final_folder_path = None  # Folder path to save images (updated dynamically)

        # Main frame for video and controls
        main_frame = Frame(self.root)
        main_frame.pack(fill=BOTH, expand=True)

        # Left Frame for Video Display
        video_frame = Frame(main_frame)
        video_frame.pack(side=LEFT, padx=10, pady=10)

        # Right Frame for Controls
        controls_frame = Frame(main_frame)
        controls_frame.pack(side=RIGHT, padx=10, pady=10, fill=Y)

        # Live camera display
        self.camera_display = Label(video_frame)
        self.camera_display.pack()

        # Controls: Camera selection
        Label(controls_frame, text="Select Camera:").pack(anchor=W, pady=5)
        self.camera_dropdown = ttk.Combobox(controls_frame, state='readonly')
        self.camera_dropdown.pack(fill=X, pady=5)
        self.camera_dropdown['values'] = self.get_available_cameras()
        self.camera_dropdown.current(0)

        self.camera_button = Button(controls_frame, text="Show Camera", command=self.show_camera)
        self.camera_button.pack(fill=X, pady=5)

        # Custom folder name entry
        self.custom_folder_entry = Entry(controls_frame)
        self.custom_folder_entry.pack(fill=X, pady=10)
        self.custom_folder_entry.insert(0, "Enter custom folder name")

        # Capture button
        self.capture_button = Button(controls_frame, text="Capture", command=self.capture_image)
        self.capture_button.pack(fill=X, pady=20)

        # Initialize session folder to save images in the current directory
        self.initialize_session_folder()

        # Start updating the camera feed
        self.update_camera()

    def get_available_cameras(self):
        index = 0
        available_cameras = []
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            else:
                available_cameras.append(index)
            cap.release()
            index += 1
        return available_cameras if available_cameras else [0]

    def show_camera(self):
        selected_index = self.camera_dropdown.get()
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(int(selected_index))

        # Set camera resolution to 720p (1280x720)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    def update_camera(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_display.imgtk = imgtk
                self.camera_display.configure(image=imgtk)
        self.root.after(10, self.update_camera)

    def initialize_session_folder(self):
        # Create a session folder with the current date and time in the current directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_folder = os.path.join(os.getcwd(), timestamp)
        os.makedirs(self.session_folder, exist_ok=True)
        self.final_folder_path = None  # Reset final folder path

    def capture_image(self):
        if not self.cap or not self.cap.isOpened():
            messagebox.showwarning("Warning", "Camera is not active!")
            return

        # Dynamically create the folder based on the textbox value (custom folder name)
        custom_folder_name = self.custom_folder_entry.get().strip()
        if custom_folder_name:
            # Update the final folder path with the new custom folder name
            self.final_folder_path = os.path.join(self.session_folder, custom_folder_name)
            os.makedirs(self.final_folder_path, exist_ok=True)
        else:
            # If no custom name, save images in the session folder
            self.final_folder_path = self.session_folder

        # Get the current frame
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showwarning("Warning", "Failed to capture image!")
            return

        # Save the captured image in the final folder
        image_filename = f"captured_image_{datetime.now().strftime('%H-%M-%S')}.jpg"
        file_path = os.path.join(self.final_folder_path, image_filename)
        cv2.imwrite(file_path, frame)
        messagebox.showinfo("Success", f"Image saved at {file_path}")

    def on_closing(self):
        if self.cap:
            self.cap.release()
        self.root.destroy()


# Initialize and run the application
root = Tk()
app = CameraApp(root)
root.protocol("WM_DELETE_WINDOW", app.on_closing)
root.mainloop()
