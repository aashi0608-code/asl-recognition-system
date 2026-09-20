from letter_inference import predict_letter, hands
from digit_inference import predict_digit

CONFIDENCE_MARGIN = 0.15
STABILITY_THRESHOLD = 25

class SignPipeline:
    def __init__(self):
        self.current_word = ""
        self.full_sentence = ""
        self.last_confirmed = None
        self.stable_sign = None
        self.stable_count = 0

    def get_prediction(self, frame, frame_rgb):
        result = hands.process(frame_rgb)
        if not result.multi_hand_landmarks:
            return None, 'none', 0.0

        letter_pred, letter_conf = predict_letter(frame_rgb)
        digit_pred, digit_conf = predict_digit(frame, frame_rgb)

        if digit_pred is not None and digit_conf > (letter_conf + CONFIDENCE_MARGIN):
            return digit_pred, 'digit', digit_conf
        else:
            return letter_pred, 'letter', letter_conf

    def process_frame(self, frame, frame_rgb):
        predicted, mode, conf = self.get_prediction(frame, frame_rgb)

        if predicted != self.stable_sign:
            self.stable_sign = predicted
            self.stable_count = 1
            self.last_confirmed = None
        else:
            self.stable_count += 1

        if (self.stable_count == STABILITY_THRESHOLD
                and predicted != self.last_confirmed
                and predicted is not None):
            if predicted == 'space':
                self.full_sentence += self.current_word + " "
                self.current_word = ""
            elif predicted == 'nothing':
                pass
            else:
                self.current_word += str(predicted)
            self.last_confirmed = predicted

        if predicted in (None, 'nothing'):
            self.last_confirmed = None

        return predicted, mode, conf

    def delete_last(self):
        if self.current_word:
            self.current_word = self.current_word[:-1]
        elif self.full_sentence:
            self.full_sentence = self.full_sentence.rstrip()

    def get_display_text(self):
        return f"{self.full_sentence}{self.current_word}"