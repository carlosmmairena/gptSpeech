import os
import tempfile
import speech_recognition as speechrecog
import openai
import whisper
from gtts import gTTS
from playsound import playsound


# Listen the voice from mic
def recognize_speech_from_mic(recognizer: speechrecog.Recognizer, microphone: speechrecog.Microphone):
    with microphone as source:
        print("Ajustando micrófono al ruido del ambiente...")
        print("Espera...")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Habla...")
        audio = recognizer.listen(source)
        return audio.get_wav_data()

# Recognize the audio from mic to whisper to get the text
def recognize_speech_with_whisper(audio_data):
    openai.api_key = os.environ["OPENAI_API_KEY"]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        temp_audio_file.write(audio_data)
        temp_audio_file.flush()

        print(temp_audio_file.name)
        model = whisper.load_model("base")
        transcript = model.transcribe(temp_audio_file.name)

    return transcript["text"]

# Send to chatgpt to get a completion
def send_to_completion(text : str):
    response = openai.Completion.create(model="text-davinci-003", prompt=text, temperature=0.5, max_tokens=100)
    text_response = response["choices"][0]["text"]
    return text_response

# Use the Google Text Voice to generate audio for the text completion and reproduce it
def text_to_voice(text : str):
    file_name = 'response.mp3'
    tts = gTTS(text, lang='es')
    tts.save(file_name)
    playsound(file_name)
    os.remove(file_name)



def main():
    recognizer = speechrecog.Recognizer()
    microphone = speechrecog.Microphone()
    is_running = True

    while(is_running):
        audio_data         = recognize_speech_from_mic(recognizer, microphone)
        audio_transcripted = recognize_speech_with_whisper(audio_data).lower()

        if audio_transcripted:
            print("Has dicho: {}".format(audio_transcripted))
            if (
                audio_transcripted.endswith("salir") or audio_transcripted.endswith("salir.") or
                audio_transcripted.endswith("fin") or audio_transcripted.endswith("fin.") or
                audio_transcripted.endswith("bye") or audio_transcripted.endswith("bye.") or
                audio_transcripted.endswith("exit") or audio_transcripted.endswith("exit.") or
                audio_transcripted.endswith("gracias") or audio_transcripted.endswith("gracias.") or
                audio_transcripted.endswith("thank you") or audio_transcripted.endswith("thank you.") or
                audio_transcripted.endswith("cerrar") or audio_transcripted.endswith("cerrar.")
            ):
                is_running = False
            else:
                response_by_ai = send_to_completion(audio_transcripted)
                print("AI Te ha respondido: {}".format(response_by_ai))
                text_to_voice(response_by_ai)
        else:
            print("No te hemos entendido...")



if __name__ == "__main__":
    main()

