"""
Attention Detection module
--------------------------
Infers attentiveness from the dominant facial emotion (via DeepFace):

  happy / neutral / surprise   -> Attentive
  sad / fear / disgust / angry -> Inattentive
  anything else                -> Distracted

Run:  python attention_detection.py     (press 'q' to quit)

Note: this file replaces the old misspelled "attentation_detection.py".
"""

import cv2
from deepface import DeepFace

ATTENTIVE_EMOTIONS = ["happy", "neutral", "surprise"]
INATTENTIVE_EMOTIONS = ["sad", "fear", "disgust", "angry"]


def attention_status(emotion):
    """Map a dominant emotion to an attentiveness status and display colour."""
    if emotion in ATTENTIVE_EMOTIONS:
        return "Attentive", (0, 255, 0)
    if emotion in INATTENTIVE_EMOTIONS:
        return "Inattentive", (0, 0, 255)
    return "Distracted", (0, 255, 255)


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
            status, color = attention_status(emotion)

            cv2.putText(frame, f"Status: {status}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"Emotion: {emotion}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        except Exception:
            pass

        cv2.imshow("Attention & Aggression Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
