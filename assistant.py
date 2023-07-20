# assistant.py
import openai
from pydub import AudioSegment
from pydub.playback import play
import os
import tempfile
import speech_recognition as sr
from gtts import gTTS
from config import WAKE_WORD, PAUSE_WORD, INITIAL_MESSAGE, GPT_MODEL
from secretkeys import OPENAI_API_KEY
import time
import ctypes
from ctypes.util import find_library

ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)

def py_error_handler(filename, line, function, err, fmt):
    pass

c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
asound = ctypes.CDLL(find_library('asound'))
asound.snd_lib_error_set_handler(c_error_handler)

openai.api_key = OPENAI_API_KEY

messages = [{"role": "system", "content": INITIAL_MESSAGE}]

def listen_and_transcribe_and_respond():
    global messages
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone(device_index=11) as source:
            print("Listening...")
            audio = recognizer.listen(source, timeout=3, phrase_time_limit=10)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            with open(wav_file.name, "wb") as file:
                print("Writing Input Audio File...")
                file.write(audio.get_wav_data())

        with open(wav_file.name, "rb") as file_to_transcribe:
            print("Transcribing Input Audio File...")
            transcript = openai.Audio.transcribe("whisper-1", file_to_transcribe)

        if check_pause_command(transcript["text"]):
            os.remove(wav_file.name)
            return "pause"

        messages.append({"role": "user", "content": transcript["text"]})

        response = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=messages
        )

        system_message = response["choices"][0]["message"]
        messages.append(system_message)

        os.remove(wav_file.name)
        return system_message["content"]

    except Exception as e:
        if wav_file:
            os.remove(wav_file.name)
        return f"An error occurred: {str(e)}"

def play_response(text):
    tts = gTTS(text, lang='en')
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_file:
        tts.save(mp3_file.name)

    audio_segment = AudioSegment.from_mp3(mp3_file.name)
    os.remove(mp3_file.name)

    play(audio_segment)

def generate_initial_response():
    global messages

    try:
        response = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=messages
        )

        system_message = response["choices"][0]["message"]
        messages.append(system_message)

        return system_message["content"]

    except Exception as e:
        return f"An error occurred: {str(e)}"

def check_wake_word(transcript):
    return WAKE_WORD in transcript.lower()

def check_pause_command(transcript):
    return PAUSE_WORD in transcript.lower()

def listen_for_wake_word_or_pause():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone(device_index=11) as source:
            print("Listening for wake word or pause command...")
            audio = recognizer.listen(source, timeout=3, phrase_time_limit=3)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            with open(wav_file.name, "wb") as file:
                print("Writing Wake Word Audio File...")
                file.write(audio.get_wav_data())

        with open(wav_file.name, "rb") as file_to_transcribe:
            print("Transcribing Wake Word Audio File...")
            transcript = openai.Audio.transcribe("whisper-1", file_to_transcribe)["text"]
            print(transcript)

        os.remove(wav_file.name)
        return transcript

    except Exception as e:
        if wav_file:
            os.remove(wav_file.name)
        print(f"An error occurred while listening for wake word or pause command: {str(e)}")
        return None

def main():
    paused = True
    wake_word_response = f"Assistant is paused. Say '{WAKE_WORD.capitalize()}' to activate the assistant."
    pause_response = f"Alright, I'm pausing. Say '{WAKE_WORD.capitalize()}' to wake me up."

    print("Assistant:", wake_word_response)
    play_response(wake_word_response)

    while True:
        transcript = listen_for_wake_word_or_pause()
        if transcript is None:
            time.sleep(1)
            continue

        if check_wake_word(transcript) and paused:
            paused = False
            initial_response = generate_initial_response()
            print("Assistant:", initial_response)
            play_response(initial_response)

        while not paused:
            response = listen_and_transcribe_and_respond()
            if response:
                if check_pause_command(response):
                    print("Assistant:", pause_response)
                    play_response(pause_response)
                    print("Pausing...")
                    paused = True
                else:
                    print("Assistant:", response)
                    play_response(response)
            else:
                time.sleep(1)
                continue

if __name__ == "__main__":
    main()  # Run the main loop

