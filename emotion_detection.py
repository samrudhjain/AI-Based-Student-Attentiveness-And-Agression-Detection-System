import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        emotion = analysis[0]['dominant_emotion']

        cv2.putText(frame, f"Emotion: {emotion}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
    except:
        pass  

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

previous_emotions = []

try:
    analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    emotion = analysis[0]['dominant_emotion']

    # Store last 5 detected emotions
    previous_emotions.append(emotion)
    if len(previous_emotions) > 5:
        previous_emotions.pop(0)

    # Count how often 'angry' appears in last 5 frames
    angry_count = previous_emotions.count('angry')

    if angry_count >= 3:  # If 3 out of 5 frames are angry, confirm aggression
        confirmed_aggression = True
    else:
        confirmed_aggression = False

except:
    emotion = "Unknown"
