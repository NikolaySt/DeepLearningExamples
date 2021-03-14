#!/usr/bin/env bash

: ${WAVEGLOW:="pretrained_models/waveglow/nvidia_waveglow256pyt_fp16.pt"}
: ${FASTPITCH:="output/FastPitch_checkpoint.pt"}
: ${BS:=32}
: ${PHRASES:="phrases/devset10.tsv"}
: ${OUTPUT_DIR:="./output/audio_$(basename ${PHRASES} .tsv)"}
: ${AMP:=true}

[ "$AMP" = true ] && AMP_FLAG="--amp"

mkdir -p "$OUTPUT_DIR"

python inference.py --cuda \
                    -i ${PHRASES} \
                    -o ${OUTPUT_DIR} \
                    --fastpitch ${FASTPITCH} \
                    --waveglow ${WAVEGLOW} \
                    --wn-channels 256 \
                    --batch-size ${BS} \
                    ${AMP_FLAG}
