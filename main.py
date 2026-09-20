import cv2
from pipeline import SignPipeline
from tts import speak

def main():
    cap = cv2.VideoCapture(0)
    pipeline = SignPipeline()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predicted, mode, conf = pipeline.process_frame(frame, frame_rgb)

        display_text = pipeline.get_display_text()
        cv2.putText(frame, display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Detected: {predicted} ({mode}, {conf:.2f})", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, "b=delete  s=speak  q=quit", (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow('ASL Recognition', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            pipeline.delete_last()
        elif key == ord('s'):
            speak(pipeline.get_display_text().strip())

    cap.release()
    cv2.destroyAllWindows()
    print("Final sentence:", pipeline.get_display_text())

if __name__ == "__main__":
    main()