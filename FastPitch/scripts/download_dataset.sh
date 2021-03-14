#!/usr/bin/env bash

set -e

DATA_DIR="dataset"
VOICE_ARCH="voice.zip"
DOWNLOAD_URL="https://orionscloud.blob.core.windows.net/bb1e7e62-03a5-4d90-b15a-abb60ad55250/Dataset/${VOICE_ARCH}"

if [ ! -d ${DATA_DIR} ]; then
  echo "Downloading ${VOICE_ARCH} ..."
  wget -q ${DOWNLOAD_URL}
  echo "Extracting ${VOICE_ARCH} ..."
  unzip ${VOICE_ARCH} -d ${DATA_DIR}
  rm -f ${VOICE_ARCH}
fi

bash ./scripts/download_tacotron2.sh
bash ./scripts/download_waveglow.sh
