from ultralytics import YOLO
import mediapipe as mp

mp_hands = mp.solutions.hands
hands_for_crop = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

model_yolo = YOLO('models/yolo_digits.pt')

def get_hand_crop(frame, frame_rgb, padding=40):
    result = hands_for_crop.process(frame_rgb)
    if not result.multi_hand_landmarks:
        return None

    h, w, _ = frame.shape
    landmarks = result.multi_hand_landmarks[0]
    xs = [lm.x * w for lm in landmarks.landmark]
    ys = [lm.y * h for lm in landmarks.landmark]

    x_min, x_max = int(min(xs)) - padding, int(max(xs)) + padding
    y_min, y_max = int(min(ys)) - padding, int(max(ys)) + padding

    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(w, x_max), min(h, y_max)

    if x_max <= x_min or y_max <= y_min:
        return None

    return frame[y_min:y_max, x_min:x_max]

def predict_digit(frame, frame_rgb):
    """Returns (prediction, confidence) or (None, 0.0) if no hand detected."""
    crop = get_hand_crop(frame, frame_rgb)
    if crop is None or crop.size == 0:
        return None, 0.0

    results = model_yolo(crop, verbose=False)
    pred = model_yolo.names[results[0].probs.top1]
    conf = results[0].probs.top1conf.item()
    return pred, conf