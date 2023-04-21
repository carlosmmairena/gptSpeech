import os
import tempfile
import speech_recognition as speechrecog
import openai
import whisper


def recognize_speech_from_mic(recognizer: speechrecog.Recognizer, microphone: speechrecog.Microphone):
    with microphone as source:
        print("Ajustando micrófono al ruido del ambiente...")
        print("Espera...")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Habla...")
        audio = recognizer.listen(source)
        return audio.get_wav_data()


def recognize_speech_with_whisper(audio_data):
    openai.api_key = os.environ["OPENAI_API_KEY"]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        temp_audio_file.write(audio_data)
        temp_audio_file.flush()

        print(temp_audio_file.name)
        model = whisper.load_model("base")
        transcript = model.transcribe(temp_audio_file.name)

    return transcript["text"]

def send_to_completion(text : str):
    response = openai.Completion.create(model="text-davinci-003", prompt=text, temperature=0.6, max_tokens=50)
    return response



def main():
    recognizer = speechrecog.Recognizer()
    microphone = speechrecog.Microphone()
    is_running = True

    while(is_running):
        audio_data         = recognize_speech_from_mic(recognizer, microphone)
        audio_transcripted = recognize_speech_with_whisper(audio_data).lower()

        if audio_transcripted:
            print("Has dicho: {}".format(audio_transcripted))
            if (audio_transcripted == "salir" or audio_transcripted == "exit" or audio_transcripted == "cerrar"):
                is_running = False
                return
            else:
                response_by_ai = send_to_completion(audio_transcripted)
                print(response_by_ai)
                #print("AI Te ha respondido: {}".format(response_by_ai))
        else:
            print("No te hemos entendido...")



if __name__ == "__main__":
    main()

