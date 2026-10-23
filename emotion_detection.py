"""
Emotion Detection module
------------------------
Classifies the dominant facial emotion in a live webcam feed using
DeepFace and displays it on screen.

Run:  python emotion_detection.py     (press 'q' to quit)
"""

import cv2
from deepface import DeepFace


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Check the camera and try again.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            analysis = DeepFace.analyze(
                frame, actions=["emotion"], enforce_detection=False
            )
            emotion = analysis[0]["dominant_emotion"]

            cv2.putText(frame, f"Emotion: {emotion}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        except Exception:
            pass

        cv2.imshow("Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
