# meal detection app
A mobile-based food detection system using YOLOv8 and React Native. This project detects Sri Lankan food items from images captured via a mobile app and returns bounding boxes, class names, and confidence scores using a Flask API backend.


🍽️ Food Detection Mobile Application using YOLOv8

This project is a mobile food detection system developed using YOLOv8, Flask, and React Native (Expo).
The system allows users to capture or upload food images from a mobile phone and automatically identifies food items with bounding boxes and confidence scores.

The YOLOv8 model is trained on a custom Sri Lankan food dataset (food detect.v1i.yolov8) and deployed via a REST API, which is consumed by a React Native frontend.

🚀 Features

📱 React Native mobile application (Android)

📸 Capture image using camera or select from gallery

🧠 YOLOv8 deep learning model for food detection

🟩 Bounding box visualization with confidence scores

🌐 Flask REST API backend

📊 Model evaluation using Precision, Recall, and mAP

🔌 Local network deployment (mobile ↔ PC)

🛠️ Tech Stack
Frontend

React Native (Expo)

JavaScript

Expo Image Picker

React Native SVG

React Navigation

Backend

Python

Flask

Flask-CORS

Ultralytics YOLOv8

OpenCV

Machine Learning

YOLOv8 Object Detection

Custom Roboflow dataset

Evaluation using mAP@0.5 and mAP@0.5:0.95
