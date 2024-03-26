import sys
import os
import time
from utils.prompt_making import make_prompt
from utils.generation import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import subprocess
import whisper

def read_text_from_file(file_path):
  with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read().strip()
  return text

# Whisper load model
model = whisper.load_model("base")

# generate text from audio
def generate_text(file_path):
  result = model.transcribe(file_path, fp16=False)
  return result

# VALL-E X download and load all models
preload_models()

# generate audio from text
def generate_voice(text_prompt, file_path):
  make_prompt(name="ja-2", audio_prompt_path="prompts/ja-2.ogg", transcript=text_prompt)
  audio_array = generate_audio(text_prompt, language="ja", prompt="ja-2")
  write_wav("vallex_generation.wav", SAMPLE_RATE, audio_array)
  # convert WAV to MP3
  subprocess.run(["ffmpeg", "-y", "-i", "vallex_generation.wav", "-b:a", "192k", file_path], input=b'y\n', check=True)

# read from file
text = read_text_from_file("talk.txt")

def generate(no, file):
  # text from multi line
  for i, line in enumerate(text.split('\n')):
    if line:
      start = time.time_ns() // 10**6
      id = "{:03d}".format(i)
      input_text = f"{id} INPUT Speech.text: {line}"
      print(input_text)
      file_path = os.path.join("result", f"{no}-{id}-vallex_generation.mp3")
      try:
        generate_voice(line, file_path)
      except Exception as e:
        print(f"Error occurred while generating voice for line {id}: {e}")
        file.write(f"{input_text}\nError: {str(e)}\n")
        continue
      result = generate_text(file_path)
      output_text = f"{id} OUTPUT Result.text: {result['text']}"
      print(output_text)
      duration = ((time.time_ns() // 10**6) - start) / 1000
      print({ "Duration": duration })
      file.write(f"{input_text}\n{output_text}\nDuration: {duration} sec\n")

count = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].strip() and sys.argv[1].strip() != "0" else 3

for i in range(count):
  no = "{:02d}".format(i)
  file_path = os.path.join("result", f"{no}-result.txt")
  with open(file_path, "a+") as file:
    generate(no, file)
  print(f"===\n {no} RESULT\n===")
  print(read_text_from_file(file_path))
