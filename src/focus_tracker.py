import cv2
import numpy as np
from tensorflow import keras
import pandas as pd
from datetime import datetime
import time

print("Loading model...")
model = keras.models.load_model('models/emotion_model.keras')
print(" Model loaded!")

emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

session_data = []
session_start = datetime.now()
blink_history = []
last_blink_time = time.time()

def calculate_eye_aspect_ratio(eye_region):
    
    h, w = eye_region.shape[:2]
    if w == 0:
        return 0
    return h / w

def analyze_blink_rate(blink_history):
   
    if len(blink_history) < 2:
        return "normal"
    
    recent_blinks = [b for b in blink_history if time.time() - b < 60]  
    blinks_per_min = len(recent_blinks)
    
    if blinks_per_min < 8:
        return "drowsy"  
    elif blinks_per_min > 25:
        return "tired"  
    else:
        return "normal"

def classify_focus_state(emotion, eyes_detected, num_eyes, blink_state, eye_positions):

    if not eyes_detected or num_eyes == 0:
        return "distracted"
    
    if num_eyes == 1:
        return "distracted"
    
    if blink_state == "drowsy":
        return "drowsy"
    
    if eye_positions:
        avg_eye_x = sum([x for x, y in eye_positions]) / len(eye_positions)
        if avg_eye_x < 0.3 or avg_eye_x > 0.7:  
            return "distracted"
    
    if emotion == 'happy':
        return "distracted"  
    elif emotion in ['neutral', 'angry']:  
        return "focused"
    elif emotion == 'sad':
        return "drowsy"
    else:
        return "neutral"

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("\n FocusTrack - Eye Tracking Enabled")
print("Press 'q' to quit and save session")
print("=" * 50)

last_log_time = time.time()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    current_emotion = "none"
    current_confidence = 0
    eyes_detected = False
    num_eyes = 0
    eye_positions = []
    blink_state = "normal"
    
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48))
        face_normalized = face_resized / 255.0
        face_input = np.expand_dims(face_normalized, axis=0)
        face_input = np.expand_dims(face_input, axis=-1)
        
        prediction = model.predict(face_input, verbose=0)
        emotion_idx = np.argmax(prediction)
        current_emotion = emotions[emotion_idx]
        current_confidence = prediction[0][emotion_idx] * 100
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
        num_eyes = len(eyes)
        eyes_detected = num_eyes > 0
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 1)
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 2)
            
            eye_center_x = (x + ex + ew//2) / frame_width
            eye_center_y = (y + ey + eh//2) / frame_height
            eye_positions.append((eye_center_x, eye_center_y))
            
            eye_region = roi_gray[ey:ey+eh, ex:ex+ew]
            ear = calculate_eye_aspect_ratio(eye_region)
            
            if ear < 0.3 and time.time() - last_blink_time > 0.3:
                blink_history.append(time.time())
                last_blink_time = time.time()
        
        blink_state = analyze_blink_rate(blink_history)
        
        focus_state = classify_focus_state(
            current_emotion, 
            eyes_detected, 
            num_eyes, 
            blink_state,
            eye_positions
        )
        
        if focus_state == "focused":
            color = (0, 255, 0)  
        elif focus_state == "distracted":
            color = (0, 0, 255)  
        elif focus_state == "drowsy":
            color = (255, 0, 255)  
        else:
            color = (255, 255, 0)  
        
        cv2.putText(frame, f"Emotion: {current_emotion} ({current_confidence:.1f}%)", 
                   (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Eyes: {num_eyes} detected", 
                   (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"State: {focus_state.upper()}", 
                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    elapsed = (datetime.now() - session_start).total_seconds()
    cv2.putText(frame, f"Session: {int(elapsed//60)}:{int(elapsed%60):02d}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    recent_blinks = len([b for b in blink_history if time.time() - b < 60])
    cv2.putText(frame, f"Blinks/min: {recent_blinks}", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Blink state: {blink_state}", 
               (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow('FocusTrack - Eye Tracking', frame)
    
    
    current_time = time.time()
    if current_time - last_log_time >= 2:
        session_data.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_seconds': int(elapsed),
            'eyes_detected': eyes_detected,
            'num_eyes': num_eyes,
            'emotion': current_emotion,
            'confidence': current_confidence,
            'blink_state': blink_state,
            'blinks_per_min': recent_blinks,
            'focus_state': focus_state if eyes_detected else 'distracted'
        })
        last_log_time = current_time
    
    frame_count += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if session_data:
    df = pd.DataFrame(session_data)
    filename = f"session_advanced_{session_start.strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    
    print("\n" + "=" * 50)
    print(" SESSION SUMMARY")
    print("=" * 50)
    print(f"Duration: {int(elapsed//60)} min {int(elapsed%60)} sec")
    print(f"Data points: {len(df)}")
    print(f"Total blinks detected: {len(blink_history)}")
    print(f"\n Focus Breakdown:")
    print(df['focus_state'].value_counts())
    print(f"\n Blink State Analysis:")
    print(df['blink_state'].value_counts())
    print(f"\n Saved: {filename}")
else:
    print("\n No data logged")