import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    if text.strip():
        engine.say(text)
        engine.runAndWait()