import torch
import torchaudio
import random
import scipy
import numpy as np

#### AUDIO TOOLS ####
standard_frame_rate = 8000
max_duration_ms = 4035
n_fft = 2048
win_length = None
hop_length = 512
n_mels = 256
n_mfcc = 256

def stereo_to_mono(audio):
    waveform, sample_rate = audio
    if waveform.shape[0] == 2:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return (waveform, sample_rate)

def resample(audio, new_sample_rate):
    waveform, sample_rate = audio

    if (sample_rate == new_sample_rate):
        return (waveform, new_sample_rate)

    resampled_waveform = torchaudio.transforms.Resample(
        sample_rate, new_sample_rate
    )(waveform)

    return (resampled_waveform, new_sample_rate)

def resize_with_silence(audio, target_ms):
    waveform, sample_rate = audio
    channel, nframes = waveform.shape

    target_nframes = sample_rate // 1000 * target_ms

    if nframes > target_nframes:
        return (waveform[:, :target_nframes], sample_rate)
    
    beginning_pad_frames = random.randint(0, target_nframes - nframes)
    ending_pad_frames = target_nframes - (nframes + beginning_pad_frames)

    beginning_pad = torch.zeros(1, beginning_pad_frames)
    ending_pad = torch.zeros(1, ending_pad_frames)

    return (torch.cat((beginning_pad, waveform, ending_pad), 1), sample_rate)
  
def audio_formatting(path, standard_frame_rate, max_duration_ms):
    audio = torchaudio.load(path)
    mono_audio = stereo_to_mono(audio)
    resampled_mono_audio = resample(mono_audio, standard_frame_rate)
    resampled_resized_mono_audio = resize_with_silence(resampled_mono_audio, max_duration_ms)

    return resampled_resized_mono_audio

mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=standard_frame_rate,
    n_mfcc=n_mfcc,
    melkwargs={
        "n_fft": n_fft,
        "n_mels": n_mels,
        "hop_length": hop_length,
        "mel_scale": "htk",
    },
)

noisy_dog_waveform, sample_rate = torchaudio.load("./dog_mic_recording.wav")
real_noise = noisy_dog_waveform[0][3000:8000].unsqueeze(0)

def add_microphone_noise(audio, real_noise=real_noise):
    waveform, sample_rate = audio

    mean = real_noise.mean()
    std = real_noise.std()
    gaussian_noise = torch.randn(noisy_dog_waveform.size())*mean + std

    frequencies, psd = scipy.signal.welch(real_noise, fs=sample_rate, nperseg=1024)

    n_frequencies = frequencies / (sample_rate / 2)
    magnitude_response = np.sqrt(psd)
    num_taps = 1025 
    fir_filter = scipy.signal.firwin2(num_taps, n_frequencies, magnitude_response[0])

    shaped_noise = torch.tensor(scipy.signal.lfilter(fir_filter, [1.0], gaussian_noise), dtype=torch.float32)
    noise_length = int(sample_rate * max_duration_ms / 1000)
    fitted_shaped_noise = shaped_noise[0][0:noise_length].unsqueeze(0)

    gain = 50000

    return (waveform + gain * fitted_shaped_noise, sample_rate)