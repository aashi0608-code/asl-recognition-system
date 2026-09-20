import numpy as np
import mediapipe as mp
import joblib

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

best_model = joblib.load('models/best_model.pkl')

def normalize_landmarks(landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    coords -= coords[0].copy()
    scale = np.linalg.norm(coords[12])
    if scale > 0:
        coords /= scale
    return coords.flatten()

def predict_letter(frame_rgb):
    #Returns (prediction, confidence) or (None, 0) if no hand detected
    result = hands.process(frame_rgb)
    if not result.multi_hand_landmarks:
        return None, 0.0

    landmarks = result.multi_hand_landmarks[0]
    row = normalize_landmarks(landmarks)

    probs = best_model.predict_proba([row])[0]
    pred = best_model.classes_[np.argmax(probs)]
    conf = np.max(probs)
    return pred, conf