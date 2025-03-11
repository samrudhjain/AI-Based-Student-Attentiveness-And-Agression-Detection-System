# ML-Based-Student-Attentiveness-And-Agression-Detection-System
📌 Machine Learning-Based Student Monitoring System
This project is a real-time Machine Learning-based Student Monitoring System that leverages computer vision and deep learning models to analyze student attentiveness and detect aggressive behavior in classrooms. The system integrates face detection, emotion recognition, and motion tracking to monitor students in real-time using pre-trained ML models.

The project consists of three core machine learning modules:

Face Detection: Identifies students’ faces in a live video feed using OpenCV’s Haar Cascade model.
Emotion Detection: Uses DeepFace (a deep learning-based model) to classify facial expressions into categories like happy, sad, angry, and neutral to determine attentiveness.
Motion Detection: Tracks movement using OpenCV’s background subtraction technique to detect sudden aggressive actions that may indicate fights.
Each module runs independently and is integrated into a main script (main.py) that combines their outputs to determine the overall status of students. If a student appears inattentive (bored, sleepy) or aggressive (angry face, rapid movement), the system can trigger alerts.

This project follows a modular approach, making it easy to extend and improve. It is built using Python, OpenCV, DeepFace, NumPy, and imutils. Future enhancements could include a web-based dashboard, notifications, and improved deep learning models for better accuracy.
