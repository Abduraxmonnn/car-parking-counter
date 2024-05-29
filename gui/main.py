import time
from datetime import datetime

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle

from sql.add_new_commer import add_car_entry
from src.utils import Park_classifier

# Decrease window size slightly
Window.size = (Window.size[0] - 50, Window.size[1] - 50)


class MainPage(Screen):
    def __init__(self, **kwargs):
        super(MainPage, self).__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', padding=50, spacing=20)

        # Add title
        title = Label(
            text="Welcome to Car Parking Detection",
            font_size='24sp',
            bold=True,
            size_hint=(1, 0.2)
        )
        layout.add_widget(title)

        # Add start button
        start_button = Button(
            text="Start Video Processing",
            font_size='20sp',
            background_color=(0.1, 0.5, 0.6, 1),  # RGBA
            color=(1, 1, 1, 1),  # Text color
            size_hint=(1, 0.3),
            pos_hint={'center_x': 0.5}
        )
        start_button.bind(on_press=self.start_video_processing)
        layout.add_widget(start_button)

        # Add quit button
        quit_button = Button(
            text="Quit",
            font_size='20sp',
            background_color=(0.6, 0.1, 0.1, 1),  # RGBA
            color=(1, 1, 1, 1),  # Text color
            size_hint=(1, 0.3),
            pos_hint={'center_x': 0.5}
        )
        quit_button.bind(on_press=self.quit_app)
        layout.add_widget(quit_button)

        self.add_widget(layout)

    def start_video_processing(self, instance):
        # Add initial entry to the database
        add_car_entry(0, 1, datetime.now())  # Assuming spot is initially empty
        self.manager.current = 'video_processing'

    def quit_app(self, instance):
        App.get_running_app().stop()


class VideoProcessingPage(Screen):
    def __init__(self, **kwargs):
        super(VideoProcessingPage, self).__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical')
        self.previous_status_list = []

        # Info layout for status texts and quit button
        info_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1), padding=10, spacing=10)

        with info_layout.canvas:
            Color(0, 0, 1, 1)  # Blue color for total spots background
            self.total_rect = Rectangle(size=(200, 40), pos=(10, 10))

            Color(0, 1, 0, 1)  # Green color for free spots background
            self.free_rect = Rectangle(size=(200, 40), pos=(220, 10))

            Color(1, 0, 0, 1)  # Red color for occupied spots background
            self.occupied_rect = Rectangle(size=(200, 40), pos=(430, 10))

        # Add spot information labels
        self.total_label = Label(
            text="Total: 0",
            color=(1, 1, 1, 1),
            bold=True,
            font_size='18sp',
            size_hint=(None, None),
            size=(200, 40)
        )
        self.free_label = Label(
            text="Free: 0",
            color=(0, 1, 0, 1),
            bold=True,
            font_size='18sp',
            size_hint=(None, None),
            size=(200, 40)
        )
        self.occupied_label = Label(
            text="Occupied: 0",
            color=(1, 0, 0, 1),
            bold=True,
            font_size='18sp',
            size_hint=(None, None),
            size=(200, 40)
        )

        info_layout.add_widget(self.total_label)
        info_layout.add_widget(self.free_label)
        info_layout.add_widget(self.occupied_label)

        # Add quit button on video processing screen
        quit_button = Button(
            text="Quit",
            font_size='16sp',
            background_color=(0.6, 0.1, 0.1, 1),
            color=(1, 1, 1, 1),
            size_hint=(None, None),
            size=(100, 50),
            pos_hint={'x': 0, 'top': 1}
        )
        quit_button.bind(on_press=self.quit_app)
        info_layout.add_widget(quit_button)

        self.layout.add_widget(info_layout)

        # Image widget for video display
        self.image_widget = Image()
        self.layout.add_widget(self.image_widget)

        self.add_widget(self.layout)

        # Define paths to video and model
        self.video_path = "../data/source/carPark small.mp4"
        self.model_path = "../data/model/trained_model.h5"
        self.car_positions_path = "../data/source/CarParkPos small"

        # Load the classifier for car park positions
        self.classifier = Park_classifier(self.car_positions_path, rect_width=90, rect_height=40)

        # Load the trained model
        self.model = load_model(self.model_path)

        # Open the video file
        self.cap = cv2.VideoCapture(self.video_path)

        # Get the frame size
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Adjust the size of the image widget to match the video frame size
        self.image_widget.size = (self.frame_width, self.frame_height)
        self.image_widget.size_hint = (None, None)

        # Initialize status list
        ret, frame = self.cap.read()
        if ret:
            self.status_list = self.initial_status_update(frame)
        else:
            self.status_list = []

        # Schedule the update function to be called periodically
        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)  # 30 FPS

    def preprocess_input_frame(self, input_frame, position, target_size=(180, 180)):
        x, y = position
        crop_frame = input_frame[y:y + 40, x:x + 90]
        resized_frame = cv2.resize(crop_frame, target_size)
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        resized_frame = np.expand_dims(resized_frame, axis=0)
        resized_frame = resized_frame / 255.0
        return resized_frame

    def initial_status_update(self, frame):
        status_list = ["Unknown"] * len(self.classifier.car_park_positions)
        for idx, pos in enumerate(self.classifier.car_park_positions):
            input_frame = self.preprocess_input_frame(frame, pos)
            predictions = self.model.predict(input_frame)
            status_list[idx] = "Occupied" if predictions[0][0] > 0.5 else "Empty"

        # Initialize previous_status_list with the same length as status_list
        self.previous_status_list = ["Unknown"] * len(self.classifier.car_park_positions)

        return status_list

    def update_frame(self, dt):
        ret, frame = self.cap.read()
        if not ret:
            return

        for idx, pos in enumerate(self.classifier.car_park_positions):
            start = pos
            end = (pos[0] + 90, pos[1] + 40)
            color = (0, 0, 255) if self.status_list[idx] == "Occupied" else (0, 255, 0)
            cv2.rectangle(frame, start, end, color, 2)
            text_position = (start[0], start[1] - 10)
            cv2.putText(frame, f'{idx + 1}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Check if the spot status has changed
            if self.status_list[idx] != self.previous_status_list[idx]:
                # Update the database based on the new spot status
                if self.previous_status_list[idx] == "Occupied" and self.status_list[idx] == "Empty":
                    # Change from occupied to empty, add new row
                    add_car_entry(0, 1, datetime.now())

        self.previous_status_list = self.status_list[:]  # Update previous status list

        free_count = self.status_list.count("Empty")
        occupied_count = self.status_list.count("Occupied")
        total_count = len(self.status_list)

        self.total_label.text = f"Total: {total_count}"
        self.free_label.text = f"Free: {free_count}"
        self.occupied_label.text = f"Occupied: {occupied_count}"

        buf = cv2.flip(frame, 0).tostring()
        image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image_widget.texture = image_texture

    def quit_app(self, instance):
        App.get_running_app().stop()


class CarParkApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(MainPage(name='main'))
        sm.add_widget(VideoProcessingPage(name='video_processing'))
        return sm


if __name__ == "__main__":
    CarParkApp().run()
