from torchaudio.pipelines import MMS_FA as bundle
from unidecode import unidecode
import re
from typing import List
import torch
import torchaudio
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model=bundle.get_model()
model.to(device)
tokenizer=bundle.get_tokenizer()
aligner=bundle.get_aligner()

def compute_alignement(wav:torch.Tensor, sentence:List[str]):
  with torch.inference_mode():
    emission,_=model(wav.to(device))
    token_spans=aligner(emission[0],tokenizer(sentence))
  return emission,token_spans


def _score(spans):
  return sum(s.score*len(s) for s in spans) /sum(len(s) for s in spans)

def preview_word(wav,spans,num_frames,transcript,sample_rate=bundle.sample_rate):
  ratio=wav.size(1)/num_frames
  x0=int(ratio*spans[0].start)
  x1=int(ratio*spans[-1].start)

  print(f"{transcript} ({_score(spans)}): {x0/sample_rate - x1/sample_rate}")


def normalize_text(text):
  text =unidecode(text.lower())
  text=re.sub("[^a-z]"," ",text)
  text=re.sub(" +"," ",text).strip()
  return text

def get_dict():
  return bundle.get_dict()

def tokenize(text):
  return tokenizer(text)

def emit_emission(spl):
  try:
    normal_transcript=normalize_text(spl['fon_transcription'])
    words=normal_transcript.split()
    # compute_alignement expects a torch.Tensor for audio and the normalized transcript string
    wave=torch.Tensor(spl['audio']["array"]).unsqueeze(0)
    emission,token_spans=compute_alignement(wave,words)

    # Calculate the total aligned duration
    aligned_duration = sum([span[0].end - span[0].start for span in token_spans])
    return {'alignment':'no_trunc'}
  except:
    return{'alignment':'trunc'}
  
