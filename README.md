# AI-Based Student Attentiveness & Aggression Detection System

A real-time computer vision system that analyses a classroom webcam feed to estimate student **attentiveness** and detect signs of **aggression**. Built with Python, OpenCV and DeepFace (deep learning).

<!-- DEMO: record main.py running (15-20 seconds), save it as demo.gif in this folder, then replace this comment with: ![Demo](demo.gif) -->

## What it does

The system combines three modules on a live video feed:

| Module | Technique | Purpose |
|---|---|---|
| Face Detection | OpenCV Haar Cascade | Locates faces in each frame |
| Emotion Detection | DeepFace (deep learning) | Classifies the dominant facial emotion |
| Motion Detection | MOG2 background subtraction | Measures how much sudden movement is happening |

**Attentiveness** is inferred from the dominant emotion:

- `happy`, `neutral`, `surprise` → **Attentive**
- `sad`, `fear`, `disgust`, `angry` → **Inattentive**
- anything else → **Distracted**

**Aggression / fight detection** needs *two* signals together, so it does not fire on a single angry face or one sudden movement:

1. High total motion in the frame (MOG2 motion above a threshold), **and**
2. A sustained angry emotion — "angry" dominant in at least **3 of the last 5** analysed frames.

## Project structure

```
.
├── main.py                  # Integrated system: all modules in one feed
├── face_detection.py        # Module demo: face detection only
├── emotion_detection.py     # Module demo: emotion recognition only
├── attention_detection.py   # Module demo: emotion -> attentiveness status
├── fight_detection.py       # Module demo: motion + emotion -> fight alert
└── requirements.txt
```

`main.py` is the full system. The other scripts run each module on its own, which is how the project was developed and tested.

## Getting started

**Requirements:** Python 3.9+ and a webcam.

```bash
# 1. Clone the repository
git clone https://github.com/samrudhjain/AI-Based-Student-Attentiveness-And-Agression-Detection-System.git
cd AI-Based-Student-Attentiveness-And-Agression-Detection-System

# 2. (Recommended) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies (DeepFace installs TensorFlow automatically)
pip install -r requirements.txt

# 4. Run the integrated system
python main.py
```

Press **`q`** to quit. The first run downloads DeepFace's pre-trained model weights, so it may take a moment to start.

To run a single module instead:

```bash
python face_detection.py
python emotion_detection.py
python attention_detection.py
python fight_detection.py
```

## Tech stack

- **Python**
- **OpenCV** — video capture, Haar Cascade face detection, MOG2 motion detection
- **DeepFace** (on TensorFlow) — facial emotion classification
- **NumPy**, **imutils** — frame handling utilities

## Limitations

Honest notes on what this is and isn't:

- Attentiveness is a **heuristic** inferred from facial emotion — it is an estimate, not a measurement of actual attention.
- Emotion analysis reads the **dominant face** in the frame; it does not yet track multiple students individually.
- Results depend on lighting and camera angle, like any webcam-based vision system.
- This is an educational / demonstration project, not a production surveillance tool.

## Future improvements

- Track and report each student individually (face IDs + per-student history)
- Web dashboard with live status and session summaries
- Alert notifications (sound / email) when aggression is confirmed
- Lighter, faster emotion model so analysis can run on more frames

## License

MIT — see [LICENSE](LICENSE).
