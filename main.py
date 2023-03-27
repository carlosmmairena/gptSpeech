import speech_recognition as speechrecog

def recognize_speech_from_mic(recognizer, microphone):
    with microphone as source:
        print("Adjusting for ambient noise, please wait...")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        print("Recognizing speech...")
        text = recognizer.recognize_google(audio)
        return text
    except speechrecog.RequestError:
        print("Error: API unavailable or unresponsive.")
    except speechrecog.UnknownValueError:
        print("Error: Unable to recognize speech.")

def main():
    recognizer = speechrecog.Recognizer()
    microphone = speechrecog.Microphone()

    text = recognize_speech_from_mic(recognizer, microphone)
    if text:
        print("You said: '{}'".format(text))
    else:
        print("No speech recognized.")

if __name__ == "__main__":
    main()

