import text2emotion as te
from transformers import pipeline
from tkinter import *
from tkinter import scrolledtext, filedialog, ttk
from pydub import AudioSegment
import os
import speech_recognition as sr
import threading


# Ensure PyTorch is installed
try:
   import torch
except ImportError:
   print("Installing PyTorch...")
   os.system('pip install torch')


# Load the emotion classifier
emotion_classifier = pipeline('sentiment-analysis', model='j-hartmann/emotion-english-distilroberta-base')


# Function for text-based emotion detection using text2emotion
def detect_emotion_text_t2e(text):
   try:
       emotions = te.get_emotion(text)
       return emotions
   except Exception as e:
       print(f"Error in text2emotion text emotion detection: {e}")
       return {}


# Function for text-based emotion detection using transformers
def detect_emotion_text_transformers(text):
   try:
       result = emotion_classifier(text)
       return result
   except Exception as e:
       print(f"Error in transformers text emotion detection: {e}")
       return []


# Function for speech-to-text and emotion detection
def detect_emotion_speech(audio_path):
   try:
       # Convert audio to WAV in case it's not in WAV format
       sound = AudioSegment.from_file(audio_path)
       wav_path = "temp_audio.wav"
       sound.export(wav_path, format="wav")


       recognizer = sr.Recognizer()
       with sr.AudioFile(wav_path) as source:
           audio = recognizer.record(source)


       # Perform speech recognition
       text = recognizer.recognize_google(audio)
       print(f"Transcribed Text: {text}")


       # Perform emotion detection on transcribed text
       result = emotion_classifier(text)
       os.remove(wav_path)  # Clean up temporary file
       return text, result
   except sr.UnknownValueError:
       return "Speech Recognition could not understand audio", None
   except sr.RequestError as e:
       return f"Could not obtain results from Google Speech Recognition service due to: {e}", None
   except Exception as e:
       print(f"Error in speech-to-text conversion: {e}")
       return None, None


# Function to display results in the GUI
def display_results():
   progress_bar.start()
   input_type = var.get()
   if input_type == "text":
       text = input_text.get("1.0", END).strip()
       if not text:
           result_text = "Please enter some text."
           output_text.delete("1.0", END)
           output_text.insert(INSERT, result_text)
           progress_bar.stop()
       else:
           threading.Thread(target=process_text_input, args=(text,)).start()
   elif input_type == "speech":
       audio_path = input_text.get("1.0", END).strip()
       if not os.path.isfile(audio_path):
           result_text = "Invalid file path. Please select a valid audio file."
           output_text.delete("1.0", END)
           output_text.insert(INSERT, result_text)
           progress_bar.stop()
       else:
           threading.Thread(target=process_speech_input, args=(audio_path,)).start()


def process_text_input(text):
   emotions_t2e = detect_emotion_text_t2e(text)
   emotions_transformers = detect_emotion_text_transformers(text)
   result_text = f"Emotions (Text - text2emotion):\n{emotions_t2e}\n\n"
   result_text += f"Emotions (Text - transformers):\n{emotions_transformers}\n"
   output_text.delete("1.0", END)
   output_text.insert(INSERT, result_text)
   progress_bar.stop()


def process_speech_input(audio_path):
   text, emotions = detect_emotion_speech(audio_path)
   if text is None and emotions is None:
       result_text = "Error in processing the audio file."
   else:
       result_text = f"Transcribed Text:\n{text}\n\nEmotions (Speech):\n{emotions}\n"
   output_text.delete("1.0", END)
   output_text.insert(INSERT, result_text)
   progress_bar.stop()


# Function to browse for a speech file
def browse_file():
   file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav *.ogg *.flac *.aac")])
   if file_path:
       input_text.delete("1.0", END)
       input_text.insert(INSERT, file_path)


# Create the main window
root = Tk()
root.title("Emotion Detection from Text and Speech - The Pycodes")


# Create and place widgets
input_label = Label(root, text="Please Enter a Text or an Audio File Path:")
input_label.pack()


input_text = scrolledtext.ScrolledText(root, wrap=WORD, width=100, height=10)
input_text.pack(padx=10, pady=10)


var = StringVar(value="text")
text_radio = Radiobutton(root, text="Text", variable=var, value="text")
text_radio.pack()
speech_radio = Radiobutton(root, text="Speech", variable=var, value="speech")
speech_radio.pack()


browse_button = Button(root, text="Browse", command=browse_file)
browse_button.pack(pady=10)


process_button = Button(root, text="Detect Emotion", command=display_results)
process_button.pack(pady=10)


output_label = Label(root, text="Output:")
output_label.pack()


output_text = scrolledtext.ScrolledText(root, wrap=WORD, width=100, height=20)
output_text.pack(padx=10, pady=10)


progress_bar = ttk.Progressbar(root, orient=HORIZONTAL, length=380, mode='indeterminate')
progress_bar.place(x=3, y=280)


# Run the GUI event loop
root.mainloop()
