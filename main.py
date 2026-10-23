"""
AI-Based Student Attentiveness & Aggression Detection System
-------------------------------------------------------------
Integrated script: runs all three modules together on one webcam feed.

  1. Face detection   - OpenCV Haar cascade
  2. Emotion analysis - DeepFace (runs every 3rd frame; it is too heavy
                        to run on every frame)
  3. Motion detection - MOG2 background subtraction

Attentiveness is inferred from the dominant emotion. A fight alert fires
only when high motion AND a sustained "angry" emotion (at least 3 of the
last 5 analysed frames) occur together, so a single angry face or a
single sudden movement does not trigger a false alarm.

Run:  python main.py     (press 'q' to quit)
"""

import cv2
import imutils
from deepface import DeepFace

# --- Attentiveness mapping (same rule as attention_detection.py) ---
ATTENTIVE_EMOTIONS = ["happy", "neutral", "surprise"]
INATTENTIVE_EMOTIONS = ["sad", "fear", "disgust", "angry"]

# --- Motion / aggression thresholds (same values as fight_detection.py) ---
MOTION_AREA_MIN = 1500      # ignore contours smaller than this
MOTION_TOTAL_LIMIT = 60000  # total motion above this counts as high motion
ANGRY_WINDOW = 5            # emotion history length
ANGRY_THRESHOLD = 3         # angry results needed to confirm aggression

ANALYZE_EVERY_N_FRAMES = 3  # run DeepFace on every Nth frame, reuse between


def attention_status(emotion):
    """Map a dominant emotion to an attentiveness status and display colour."""
    if emotion in ATTENTIVE_EMOTIONS:
        return "Attentive", (0, 255, 0)
    if emotion in INATTENTIVE_EMOTIONS:
        return "Inattentive", (0, 0, 255)
    return "Distracted", (0, 255, 255)


def aggression_confirmed(emotion_history):
    """Aggression needs a sustained angry signal, not a single frame."""
    return emotion_history.count("angry") >= ANGRY_THRESHOLD


def main():
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    fgbg = cv2.createBackgroundSubtractorMOG2()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Check the camera and try again.")
        return

    emotion = "Unknown"
    emotion_history = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_count += 1

        # 1) Face detection
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 2) Motion detection (MOG2 foreground mask)
        fgmask = fgbg.apply(gray)
        contours, _ = cv2.findContours(
            fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        total_motion = 0
        for contour in contours:
            if cv2.contourArea(contour) > MOTION_AREA_MIN:
                total_motion += cv2.contourArea(contour)
        motion_detected = total_motion > MOTION_TOTAL_LIMIT

        # 3) Emotion analysis (every Nth frame only)
        if frame_count % ANALYZE_EVERY_N_FRAMES == 0:
            try:
                analysis = DeepFace.analyze(
                    frame, actions=["emotion"], enforce_detection=False
                )
                emotion = analysis[0]["dominant_emotion"]
                emotion_history.append(emotion)
                if len(emotion_history) > ANGRY_WINDOW:
                    emotion_history.pop(0)
            except Exception:
                emotion = "Unknown"

        # Combined status
        status, color = attention_status(emotion)
        fight = motion_detected and aggression_confirmed(emotion_history)

        if fight:
            cv2.putText(frame, "Fight Detected!", (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f"Status: {status}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Emotion: {emotion}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        cv2.imshow("Attentiveness & Aggression Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
