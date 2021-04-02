import torch
import numpy as np
from scipy.io.wavfile import write
from hubconf import nvidia_tacotron2, nvidia_fastpitch, nvidia_waveglow
from scipy.interpolate import CubicHermiteSpline

sampling_rate = 22050


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")


device = get_device()

waveglow = nvidia_waveglow()
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow.eval()
waveglow.training = False
waveglow.to(device)

tacotron2 = nvidia_tacotron2()
tacotron2.eval()
tacotron2.training = False
tacotron2.to(device)

fastpitch = nvidia_fastpitch(device, model_math="fp16")
fastpitch.eval()
fastpitch.training = False
fastpitch.to(device)

denoiser = fastpitch.denoiser(waveglow).to(device)

batch = tacotron2.text_to_batch([
    "Facebook is showing information to help you better understand the purpose of this event."
], device, "english_basic", ["english_cleaners"])


def fastpitch_infer(output_filename="f_audio.wav",
                    pitch=None,
                    pitch_transform=None):
    gen_kw = {
        'pace': 2.0,
        'speaker': 0,
        'pitch_tgt': pitch,
        'pitch_transform': pitch_transform
    }

    with torch.no_grad():
        mel, mel_lens, *_ = fastpitch.infer(batch['text'], batch['text_lens'],
                                            **gen_kw)
        audios = waveglow.infer(mel)
        audios = denoiser(audios.float(), strength=0.01).squeeze(1)

    audios = fastpitch.post_process(audios, mel_lens)

    write(output_filename, sampling_rate, audios[0].data.cpu().numpy())
    torch.cuda.empty_cache()


def tacotron_infer(output_filename="t_audio.wav"):
    with torch.no_grad():
        mel, mel_lens, *_ = tacotron2.infer(batch['text'], batch['text_lens'])
        audios = waveglow.infer(mel)
        audios = denoiser(audios.float(), strength=0.01).squeeze(1)

    audios = tacotron2.post_process(audios, mel_lens)

    write(output_filename, sampling_rate, audios[0].data.cpu().numpy())
    torch.cuda.empty_cache()


def get_custom_pitch(length, keypoints):

    x = np.arange(1, keypoints.shape[0] + 1, 1)   # x coordinates of turning points
    xx = np.linspace(x.min(), x.max(), length)
    y = np.array(keypoints)  # y coordinates of turning points

    cspline = CubicHermiteSpline(x=x, y=y,
                                 dydx=np.zeros_like(y))  # interpolator
    pitch = torch.Tensor(cspline(xx)).unsqueeze(0)
    print(pitch.shape)
    print(pitch)
    return pitch


#fastpitch_infer("f_audio_1.wav",
#                pitch_transform=fastpitch.build_pitch_transformation())

pitch = get_custom_pitch(88, np.array([1, 0.2, 0.3, 0, 0]))
fastpitch_infer("f_audio_2.wav", pitch=pitch)

#tacotron_infer()
