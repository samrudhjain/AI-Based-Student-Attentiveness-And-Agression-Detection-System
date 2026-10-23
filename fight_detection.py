"""
Fight / Aggression Detection module
-----------------------------------
Combines two signals from a live webcam feed:

  1. Motion  - MOG2 background subtraction; a large amount of sudden
               movement counts as high motion.
  2. Emotion - DeepFace; aggression is confirmed only when "angry" is
               the dominant emotion in at least 3 of the last 5
               analysed frames.

A fight is reported only when BOTH signals fire together, which avoids
false alarms from a single angry face or a single sudden movement.

Run:  python fight_detection.py     (press 'q' to quit)
"""

import cv2
import imutils
from deepface import DeepFace

MOTION_AREA_MIN = 1500      # ignore contours smaller than this
MOTION_TOTAL_LIMIT = 60000  # total motion above this counts as high motion
ANGRY_WINDOW = 5            # emotion history length
ANGRY_THRESHOLD = 3         # angry results needed to confirm aggression


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Check the camera and try again.")
        return

    fgbg = cv2.createBackgroundSubtractorMOG2()
    previous_emotions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=600)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(gray)

        contours, _ = cv2.findContours(
            fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        total_motion = 0
        for contour in contours:
            if cv2.contourArea(contour) > MOTION_AREA_MIN:
                total_motion += cv2.contourArea(contour)

        motion_detected = total_motion > MOTION_TOTAL_LIMIT

        try:
            analysis = DeepFace.analyze(
                frame, actions=["emotion"], enforce_detection=False
            )
            emotion = analysis[0]["dominant_emotion"]

            previous_emotions.append(emotion)
            if len(previous_emotions) > ANGRY_WINDOW:
                previous_emotions.pop(0)

            angry_count = previous_emotions.count("angry")
            confirmed_aggression = angry_count >= ANGRY_THRESHOLD

        except Exception:
            emotion = "Unknown"
            confirmed_aggression = False

        if motion_detected and confirmed_aggression:
            status = "Fight Detected!"
            color = (0, 0, 255)  # Red
        else:
            status = "Normal"
            color = (0, 255, 0)  # Green

        cv2.putText(frame, f"Status: {status}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Emotion: {emotion}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        cv2.imshow("Fight Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
