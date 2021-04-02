import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import urllib.request
import os
import sys


def _load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt['state_dict']
    if _checkpoint_from_distributed(state_dict):
        state_dict = _unwrap_distributed(state_dict)

    return state_dict, ckpt['config']


def _text_to_batch(lines,
                   device,
                   symbol_set="english_basic",
                   text_cleaners=["english_cleaners"],
                   batch_size=128):

    from common.text.text_processing import TextProcessing

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


def _build_pitch_transformation(custom=False,
                                flatten=False,
                                invert=False,
                                amplify=1.0,
                                shift_hz=0.0):

    if custom:

        def custom_(pitch, pitch_lens, mean, std):
            from FastPitch.pitch_transform import pitch_transform_custom
            return (pitch_transform_custom(pitch * std + mean, pitch_lens) -
                    mean) / std

        return custom_

    fun = 'pitch'
    if flatten:
        fun = f'({fun}) * 0.0'
    if invert:
        fun = f'({fun}) * -1.0'
    if amplify:
        fun = f'({fun}) * {amplify}'
    if shift_hz != 0.0:
        fun = f'({fun}) + {shift_hz} / std'
    return eval(f'lambda pitch, pitch_lens, mean, std: {fun}')


def _denoiser(waveglow,
              filter_length=1024,
              n_overlap=4,
              win_length=1024,
              mode='zeros'):
    from FastPitch.waveglow.denoiser import Denoiser
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


def _checkpoint_from_distributed(state_dict):
    """
    Checks whether checkpoint was generated by DistributedDataParallel. DDP
    wraps model in additional "module.", it needs to be unwrapped for single
    GPU inference.
    :param state_dict: model's state dict
    """
    ret = False
    for key, _ in state_dict.items():
        if key.find('module.') != -1:
            ret = True
            break
    return ret


def _unwrap_distributed(state_dict):
    """
    Unwraps model from DistributedDataParallel.
    DDP wraps model in additional "module.", it needs to be removed for single
    GPU inference.
    :param state_dict: model's state dict
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.1.', '')
        new_key = new_key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict


def _download_checkpoint(checkpoint, force_reload):
    model_dir = os.path.join(torch.hub._get_torch_home(), 'checkpoints')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    ckpt_file = os.path.join(model_dir, os.path.basename(checkpoint))
    if not os.path.exists(ckpt_file) or force_reload:
        sys.stderr.write('Downloading checkpoint from {}\n'.format(checkpoint))
        urllib.request.urlretrieve(checkpoint, ckpt_file)
    return ckpt_file


def nvidia_fastpitch(pretrained=True, **kwargs):
    """Constructs a FastPitch  model (nn.module with additional infer(input) method).
    For detailed information on model input and output, training recipies, inference and performance
    visit: github.com/NVIDIA/DeepLearningExamples and/or ngc.nvidia.com

    Args (type[, default value]):
        pretrained (bool, True): If True, returns a model pretrained on LJ Speech dataset.
        model_math (str, 'fp32'): returns a model in given precision ('fp32' or 'fp16')
    """

    from FastPitch import models
    from types import SimpleNamespace 

    fp16 = "model_math" in kwargs and kwargs["model_math"] == "fp16"
    force_reload = "force_reload" in kwargs and kwargs["force_reload"]

    if pretrained:
        if fp16:
            checkpoint = 'https://orionscloud.blob.core.windows.net/bb1e7e62-03a5-4d90-b15a-abb60ad55250/Checkpoints/nvidia_fastpitch_fp16_20210323.pt'
        else:
            raise Exception("pretrained model is available only for fp16. Set a param model_math=fp16.")
        ckpt_file = _download_checkpoint(checkpoint, force_reload)
        state_dict, config = _load_checkpoint(ckpt_file)
    else:
        config = {
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
        for k, v in kwargs.items():
            if k in config.keys():
                config[k] = v

    if not "symbol_set" in config.keys():
        config["symbol_set"] = "english_basic"

    config_args = SimpleNamespace(**config)
    model_config = models.get_model_config("FastPitch", config_args)

    m = models.get_model("FastPitch",
                         model_config,
                         forward_is_infer=False,
                         jitable=False)

    if pretrained:
        m.load_state_dict(state_dict, strict=True)

    m.denoiser = _denoiser
    m.text_to_batch = _text_to_batch
    m.post_process = _post_process
    m.build_pitch_transformation = _build_pitch_transformation

    return m


def nvidia_tacotron2(pretrained=True, **kwargs):
    """Constructs a Tacotron 2 model (nn.module with additional infer(input) method).
    For detailed information on model input and output, training recipies, inference and performance
    visit: github.com/NVIDIA/DeepLearningExamples and/or ngc.nvidia.com

    Args (type[, default value]):
        pretrained (bool, True): If True, returns a model pretrained on LJ Speech dataset.
        model_math (str, 'fp32'): returns a model in given precision ('fp32' or 'fp16')
        n_symbols (int, 148): Number of symbols used in a sequence passed to the prenet, see
                              https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/tacotron2/text/symbols.py
        p_attention_dropout (float, 0.1): dropout probability on attention LSTM (1st LSTM layer in decoder)
        p_decoder_dropout (float, 0.1): dropout probability on decoder LSTM (2nd LSTM layer in decoder)
        max_decoder_steps (int, 1000): maximum number of generated mel spectrograms during inference
    """

    from Tacotron2.tacotron2 import model as tacotron2
    from Tacotron2.models import lstmcell_to_float, batchnorm_to_float

    fp16 = "model_math" in kwargs and kwargs["model_math"] == "fp16"
    force_reload = "force_reload" in kwargs and kwargs["force_reload"]

    if pretrained:
        if fp16:
            checkpoint = 'https://api.ngc.nvidia.com/v2/models/nvidia/tacotron2_pyt_ckpt_amp/versions/19.09.0/files/nvidia_tacotron2pyt_fp16_20190427'
        else:
            checkpoint = 'https://api.ngc.nvidia.com/v2/models/nvidia/tacotron2_pyt_ckpt_fp32/versions/19.09.0/files/nvidia_tacotron2pyt_fp32_20190427'
        ckpt_file = _download_checkpoint(checkpoint, force_reload)
        state_dict, config = _load_checkpoint(ckpt_file)
    else:
        config = {
            'mask_padding': False,
            'n_mel_channels': 80,
            'n_symbols': 148,
            'symbols_embedding_dim': 512,
            'encoder_kernel_size': 5,
            'encoder_n_convolutions': 3,
            'encoder_embedding_dim': 512,
            'attention_rnn_dim': 1024,
            'attention_dim': 128,
            'attention_location_n_filters': 32,
            'attention_location_kernel_size': 31,
            'n_frames_per_step': 1,
            'decoder_rnn_dim': 1024,
            'prenet_dim': 256,
            'max_decoder_steps': 1000,
            'gate_threshold': 0.5,
            'p_attention_dropout': 0.1,
            'p_decoder_dropout': 0.1,
            'postnet_embedding_dim': 512,
            'postnet_kernel_size': 5,
            'postnet_n_convolutions': 5,
            'decoder_no_early_stopping': False
        }
        for k, v in kwargs.items():
            if k in config.keys():
                config[k] = v

    m = tacotron2.Tacotron2(**config)

    if fp16:
        m = batchnorm_to_float(m.half())
        m = lstmcell_to_float(m)

    if pretrained:
        m.load_state_dict(state_dict)

    m.text_to_batch = _text_to_batch
    m.denoiser = _denoiser
    m.text_to_batch = _text_to_batch
    m.post_process = _post_process

    return m


def nvidia_waveglow(pretrained=True, **kwargs):
    """Constructs a WaveGlow model (nn.module with additional infer(input) method).
    For detailed information on model input and output, training recipies, inference and performance
    visit: github.com/NVIDIA/DeepLearningExamples and/or ngc.nvidia.com

    Args:
        pretrained (bool): If True, returns a model pretrained on LJ Speech dataset.
        model_math (str, 'fp32'): returns a model in given precision ('fp32' or 'fp16')
    """

    from Tacotron2.waveglow import model as waveglow
    from Tacotron2.models import batchnorm_to_float

    fp16 = "model_math" in kwargs and kwargs["model_math"] == "fp16"
    force_reload = "force_reload" in kwargs and kwargs["force_reload"]

    if pretrained:
        if fp16:
            checkpoint = 'https://api.ngc.nvidia.com/v2/models/nvidia/waveglow_ckpt_amp/versions/19.09.0/files/nvidia_waveglowpyt_fp16_20190427'
        else:
            checkpoint = 'https://api.ngc.nvidia.com/v2/models/nvidia/waveglow_ckpt_fp32/versions/19.09.0/files/nvidia_waveglowpyt_fp32_20190427'
        ckpt_file = _download_checkpoint(checkpoint, force_reload)
        state_dict, config = _load_checkpoint(ckpt_file)
    else:
        config = {
            'n_mel_channels': 80,
            'n_flows': 12,
            'n_group': 8,
            'n_early_every': 4,
            'n_early_size': 2,
            'WN_config': {
                'n_layers': 8,
                'kernel_size': 3,
                'n_channels': 512
            }
        }
        for k, v in kwargs.items():
            if k in config.keys():
                config[k] = v
            elif k in config['WN_config'].keys():
                config['WN_config'][k] = v

    m = waveglow.WaveGlow(**config)

    if fp16:
        m = batchnorm_to_float(m.half())
        for mat in m.convinv:
            mat.float()

    if pretrained:
        m.load_state_dict(state_dict)

    return m
