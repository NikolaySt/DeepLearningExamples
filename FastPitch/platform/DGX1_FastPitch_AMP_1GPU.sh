#!/bin/bash

mkdir -p output
python train.py \
    --amp \
    --cuda \
    -o ./output/ \
    --log-file output/nvlog.json \
    --dataset-path dataset \
    --training-files dataset/mel_dur_pitch_text_train_filelist.txt \
    --validation-files dataset/mel_dur_pitch_text_test_filelist.txt \
    --pitch-mean-std-file dataset/pitch_char_stats__audio_text_train_filelist.json \
    --epochs 1500 \
    --epochs-per-checkpoint 100 \
    --warmup-steps 1000 \
    -lr 0.1 \
    -bs 64 \
    --optimizer lamb \
    --grad-clip-thresh 1000.0 \
    --dur-predictor-loss-scale 0.1 \
    --pitch-predictor-loss-scale 0.1 \
    --weight-decay 1e-6 \
    --gradient-accumulation-steps 4
