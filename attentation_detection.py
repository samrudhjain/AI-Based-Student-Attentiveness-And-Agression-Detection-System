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

        if emotion in ['happy', 'neutral', 'surprise']:
            status = "Attentive"
            color = (0, 255, 0)  
        elif emotion in ['sad', 'fear', 'disgust', 'angry']:
            status = "Inattentive"
            color = (0, 0, 255)  
        else:
            status = "Distracted"
            color = (0, 255, 255)  

        
        cv2.putText(frame, f"Status: {status}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Emotion: {emotion}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, color, 2, cv2.LINE_AA)
    except:
        pass  

    cv2.imshow('Attention & Aggression Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
