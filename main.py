import os
import tempfile
import speech_recognition as speechrecog
import openai


def recognize_speech_from_mic(recognizer: speechrecog.Recognizer, microphone: speechrecog.Microphone):
    with microphone as source:
        print("Adjusting for ambient noise, please wait...")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Listening...")
        audio = recognizer.listen(source)
        return audio.get_wav_data()

def recognize_speech_with_whisper(audio_data):
    openai.api_key = os.environ["OPENAI_API_KEY"]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
        temp_audio_file.write(audio_data)
        temp_audio_file.flush()

        print(temp_audio_file.name)
        transcript = openai.Audio.transcribe("whisper-1", temp_audio_file)

        #transcript = openai.Audio.transcribe(
        #    file = temp_audio_file,
        #    model = "whisper-1",
        #    file_format = "wav"
        #)

    return transcript


def main():
    recognizer = speechrecog.Recognizer()
    microphone = speechrecog.Microphone()

    audio_data = recognize_speech_from_mic(recognizer, microphone)
    text = recognize_speech_with_whisper(audio_data)

    if text:
        print("You said: '{}'".format(text))
    else:
        print("No speech recognized.")

if __name__ == "__main__":
    main()

