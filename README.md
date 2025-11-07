🧠 FaceFusion — AI-Powered Face Averager

FaceFusion is my first hands-on project using Python, OpenCV, and MediaPipe to explore computer vision and facial landmark detection.
This tool takes multiple face images, detects key facial landmarks, aligns them, and blends them together to generate an average face — a single composite image that represents the combined features of all inputs.

✨ Features
🧩 Face detection and landmark extraction using MediaPipe Face Mesh
🎨 Face alignment, warping, and blending with OpenCV
⚡ Automatic normalization of image size and shape
💾 Generates a merged my_average_face.jpg as the final result

🧰 Tech Stack
Python 🐍
OpenCV
MediaPipe
NumPy

🚀 How to Run
pip install --upgrade pip
pip install opencv-python mediapipe numpy
python facefusion.py

Or try it directly on Google Colab — no setup needed!

💬 Notes
This is my first time experimenting with OpenCV and machine learning–based image processing, so the merging quality isn’t perfect yet — but I’m working on improving it soon! 😊
