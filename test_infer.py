import torch
from scipy.io.wavfile import write
from hubconf import nvidia_tacotron2, nvidia_fastpitch, nvidia_waveglow

device = 'cpu'
waveglow = nvidia_waveglow()

waveglow = waveglow.remove_weightnorm(waveglow)
waveglow.eval()
waveglow.training = False
waveglow.to(device)

tacotron2 = nvidia_tacotron2()
tacotron2.eval()
tacotron2.training = False
tacotron2.to(device)

fastpitch = nvidia_fastpitch(device)

denoiser = fastpitch.denoiser(waveglow).to(device)

batch = tacotron2.text_to_batch([
    "Facebook is showing information to help you better understand the purpose of this event.",
    "She sells seashells by the seashore, shells she sells are great"
], device, "english_basic", ["english_cleaners"])

gen_kw = {
    'pace': 1.0,
    'speaker': 0,
    'pitch_tgt': None,
    'pitch_transform': fastpitch.build_pitch_transformation()
}

with torch.no_grad():
    mel, mel_lens, *_ = fastpitch(batch['text'], batch['text_lens'], **gen_kw)
    audios = waveglow.infer(mel)
    audios = denoiser(audios.float(), strength=0.01).squeeze(1)
    audio_data = audios[0].data.cpu().numpy()

sampling_rate = 22050
write("f_audio.wav", sampling_rate, audio_data)
torch.cuda.empty_cache()

with torch.no_grad():
    mel, mel_lens, *_  = tacotron2.infer(batch['text'], batch['text_lens'])
    audios = waveglow.infer(mel)
    audios = denoiser(audios.float(), strength=0.01).squeeze(1)
    audio_data = audios[0].data.cpu().numpy()

sampling_rate = 22050
write("t_audio.wav", sampling_rate, audio_data)
torch.cuda.empty_cache()