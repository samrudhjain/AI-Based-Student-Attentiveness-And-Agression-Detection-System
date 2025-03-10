import cv2
import imutils
from deepface import DeepFace

cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2()

previous_emotions = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=600)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_motion = 0 
    for contour in contours:
        if cv2.contourArea(contour) > 1500:  
            total_motion += cv2.contourArea(contour)

    motion_detected = total_motion > 50000  

    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = analysis[0]['dominant_emotion']

        previous_emotions.append(emotion)
        if len(previous_emotions) > 5:
            previous_emotions.pop(0)

        angry_count = previous_emotions.count('angry')
        confirmed_aggression = angry_count >= 3 

    except:
        emotion = "Unknown"
        confirmed_aggression = False

    if motion_detected and confirmed_aggression:
        status = "Fight Detected! 🚨"
        color = (0, 0, 255)  # Red
    else:
        status = "Normal"
        color = (0, 255, 0)  # Green

    cv2.putText(frame, f"Status: {status}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Emotion: {emotion}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                1, color, 2, cv2.LINE_AA)

    cv2.imshow('Fight Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
