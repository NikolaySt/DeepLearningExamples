#!/usr/bin/env bash

set -e

DATA_DIR="LJSpeech-1.1"
LJS_ARCH="JDVoice_HarryPotter.zip"
LJS_URL="https://orionscloud.blob.core.windows.net/bb1e7e62-03a5-4d90-b15a-abb60ad55250/Dataset/${LJS_ARCH}"

if [ ! -d ${DATA_DIR} ]; then
  echo "Downloading ${LJS_ARCH} ..."
  wget -q ${LJS_URL}
  echo "Extracting ${LJS_ARCH} ..."
  unzip ${LJS_ARCH}
  rm -f ${LJS_ARCH}
fi

bash ./scripts/download_tacotron2.sh
bash ./scripts/download_waveglow.sh
