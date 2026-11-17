
import wave
from pathlib import Path
import numpy as np
import torch
SR=16000;N_FFT=512;HOP=160;BINS=257
HERE=Path(__file__).resolve().parent;SOURCE=HERE.parent;ROOT=HERE.parents[3];OUT=ROOT/'outputs/5.4'
def read_wave(path):
 with wave.open(str(path),'rb') as f:
  assert f.getnchannels()==1 and f.getframerate()==SR and f.getsampwidth()==2
  raw=f.readframes(f.getnframes())
 return np.frombuffer(raw,dtype='<i2').astype('float32')/32768.
def write_wave(path,values):
 values=np.asarray(values,dtype='float32');clipped=float(np.mean(np.abs(values)>1));data=np.round(np.clip(values,-1,1)*32767).astype('<i2')
 with wave.open(str(path),'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(SR);f.writeframes(data.tobytes())
 return clipped
def magnitude(waveform):
 return torch.stft(waveform.float(),n_fft=N_FFT,hop_length=HOP,win_length=N_FFT,window=torch.hann_window(N_FFT,device=waveform.device),return_complex=True,center=True).abs()
