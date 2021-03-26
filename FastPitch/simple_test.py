import models
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np
from scipy.io.wavfile import write
from torch.nn.utils.rnn import pad_sequence

from common.text.text_processing import TextProcessing
from pitch_transform import pitch_transform_custom
from waveglow import model as glow
from waveglow.denoiser import Denoiser

sys.modules['glow'] = glow

config = {
    "output": "output/",
    "log_file": None,
    "cuda": False,
    "cudnn_benchmark": False,
    "sigma_infer": 0.9,
    "denoising_strength": 0.01,
    "sampling_rate": 22050,
    "stft_hop_length": 256,
    "include_warmup": False,
    "repeats": 1,
    "speaker": 0,
    "fade_out": 10,
    "pace": 1.0,
    "pitch_transform_flatten": False,
    "pitch_transform_invert": False,
    "pitch_transform_amplify": 1.0,
    "pitch_transform_shift": 0.0,
    "pitch_transform_custom": False,
    "text_cleaners": ["english_cleaners"],
    "symbol_set": "english_basic"
}
fastpitch_config = {
    "n_speakers": 1,
    "symbol_set": "english_basic",
    "n_mel_channels": 80,
    "max_seq_len": 2048,
    "n_symbols": 148,
    "padding_idx": 0,
    "symbols_embedding_dim": 384,
    "in_fft_n_layers": 6,
    "in_fft_n_heads": 1,
    "in_fft_d_head": 64,
    "in_fft_conv1d_kernel_size": 3,
    "in_fft_conv1d_filter_size": 1536,
    "in_fft_output_size": 384,
    "p_in_fft_dropout": 0.1,
    "p_in_fft_dropatt": 0.1,
    "p_in_fft_dropemb": 0.0,
    "out_fft_n_layers": 6,
    "out_fft_n_heads": 1,
    "out_fft_d_head": 64,
    "out_fft_conv1d_kernel_size": 3,
    "out_fft_conv1d_filter_size": 1536,
    "out_fft_output_size": 384,
    "p_out_fft_dropout": 0.1,
    "p_out_fft_dropatt": 0.1,
    "p_out_fft_dropemb": 0.0,
    "dur_predictor_kernel_size": 3,
    "dur_predictor_filter_size": 256,
    "p_dur_predictor_dropout": 0.1,
    "dur_predictor_n_layers": 2,
    "pitch_predictor_kernel_size": 3,
    "pitch_predictor_filter_size": 256,
    "p_pitch_predictor_dropout": 0.1,
    "pitch_predictor_n_layers": 2,
    "pitch_embedding_kernel_size": 3,
    "speaker_emb_weight": 1.0
}
args = SimpleNamespace(**config)
fastpitch_arg = SimpleNamespace(**fastpitch_config)


def build_model(model_args, device):
    model_name = "FastPitch"

    model_config = models.get_model_config(model_name, model_args)

    model = models.get_model(model_name,
                             model_config,
                             device,
                             forward_is_infer=True,
                             jitable=False)
    return model


def load_checkpoint(model, path):
    if path == '' or path == None:
        return
    checkpoint_data = torch.load(path, map_location="cpu")
    status = ''

    sd = checkpoint_data['state_dict']

    if any(key.startswith('module.') for key in sd):
        sd = {k.replace('module.', ''): v for k, v in sd.items()}

    status += ' ' + str(model.load_state_dict(sd, strict=True))

    print(f'Loaded {path}{status}')

def _text_to_batch(lines, device, symbol_set = "english_basic", text_cleaners = ["english_cleaners"], batch_size=128):
    columns = ['text']
    fields = [lines]
    fields = {c: f for c, f in zip(columns, fields)}

    tp = TextProcessing(symbol_set, text_cleaners)

    fields['text'] = [
        torch.LongTensor(tp.encode_text(text)) for text in fields['text']
    ]
    order = np.argsort([-t.size(0) for t in fields['text']])

    fields['text'] = [fields['text'][i] for i in order]
    fields['text_lens'] = torch.LongTensor([t.size(0) for t in fields['text']])

    # cut into batches & pad
    batches = []
    for b in range(0, len(order), batch_size):
        batch = {f: values[b:b + batch_size] for f, values in fields.items()}
        for f in batch:
            if f == 'text':
                batch[f] = pad_sequence(batch[f], batch_first=True)
            if type(batch[f]) is torch.Tensor:
                batch[f] = batch[f].to(device)
        batches.append(batch)

    return batches[0]


def build_pitch_transformation(args):
    if args.pitch_transform_custom:

        def custom_(pitch, pitch_lens, mean, std):
            return (pitch_transform_custom(pitch * std + mean, pitch_lens) -
                    mean) / std

        return custom_

    fun = 'pitch'
    if args.pitch_transform_flatten:
        fun = f'({fun}) * 0.0'
    if args.pitch_transform_invert:
        fun = f'({fun}) * -1.0'
    if args.pitch_transform_amplify:
        ampl = args.pitch_transform_amplify
        fun = f'({fun}) * {ampl}'
    if args.pitch_transform_shift != 0.0:
        hz = args.pitch_transform_shift
        fun = f'({fun}) + {hz} / std'
    return eval(f'lambda pitch, pitch_lens, mean, std: {fun}')


def load_waveglow(device):
    waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub',
                              'nvidia_waveglow',
                              map_location=device)
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.eval()
    waveglow.training = False
    waveglow.to(device)
    return waveglow


def _denoiser(waveglow,
              filter_length=1024,
              n_overlap=4,
              win_length=1024,
              mode='zeros'):
    denoiser = Denoiser(waveglow, filter_length, n_overlap, win_length, mode)
    return denoiser


def _post_process(audios, mel_lens, fade_out=10, stft_hop_length=256):
    result = []
    for i, audio in enumerate(audios):
        audio = audio[:mel_lens[i].item() * stft_hop_length]

        if fade_out:
            fade_len = fade_out * stft_hop_length
            fade_w = torch.linspace(1.0, 0.0, fade_len)
            audio[-fade_len:] *= fade_w.to(audio.device)

        audio = audio / torch.max(torch.abs(audio))
        result.append(audio)
    return result


def fastpitch_model(device, checkpoint=None):
    fastpitch_arg = SimpleNamespace(**fastpitch_config)

    fastpitch = build_model(fastpitch_arg, device)
    if checkpoint is not None:
        load_checkpoint(fastpitch, checkpoint)

    fastpitch.denoiser = _denoiser
    fastpitch.text_to_batch = _text_to_batch
    fastpitch.post_process = _post_process

    fastpitch.eval()
    return fastpitch


def main():
    device = torch.device('cuda' if args.cuda else 'cpu')
    model = fastpitch_model(device, "output/FastPitch_checkpoint.pt").to(device)

    batch = model.text_to_batch([
        "Facebook is showing information to help you better understand the purpose of this event."
    ], device, args.symbol_set, args.text_cleaners)

    waveglow = load_waveglow(device)
    
    denoiser = model.denoiser(waveglow).to(device)

    gen_kw = {
        'pace': args.pace,
        'speaker': args.speaker,
        'pitch_tgt': None,
        'pitch_transform': build_pitch_transformation(args)
    }

    with torch.no_grad():
        mel, mel_lens, *_ = model(batch['text'], batch['text_lens'], **gen_kw)
        audios = waveglow.infer(mel)
        audios = denoiser(audios.float(),
                          strength=args.denoising_strength).squeeze(1)

    result = model.post_process(audios, mel_lens)

    for i, audio in enumerate(result):
        fname = f'audio_{i}.wav'
        audio_path = Path(args.output, fname)
        write(audio_path, args.sampling_rate, audio.cpu().numpy())


if __name__ == '__main__':
    main()




# device = "cpu"
# model = nvidia_fastpitch(device=device)

# model.eval()

# batch = model.text_to_batch([
#     "Facebook is showing information to help you better understand the purpose of this event."
# ], device)

# waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub',
#                           'nvidia_waveglow',
#                           map_location=device)
# waveglow = waveglow.remove_weightnorm(waveglow)
# waveglow.eval()
# waveglow.training = False
# waveglow.to(device)

# denoiser = model.denoiser(waveglow).to(device)

# gen_kw = {
#     'pace': 1.0,
#     'speaker': 0,
#     'pitch_tgt': None,
#     'pitch_transform': model.build_pitch_transformation()
# }

# with torch.no_grad():
#     mel, mel_lens, *_ = model(batch['text'], batch['text_lens'], **gen_kw)
#     audios = waveglow.infer(mel)
#     audios = denoiser(audios.float(), strength=0.01).squeeze(1)

# result = model.post_process(audios, mel_lens)

# for i, audio in enumerate(result):
#     audio_path = f'audio_{i}.wav'
#     from scipy.io.wavfile import write
#     write(audio_path, 22050, audio.cpu().numpy())