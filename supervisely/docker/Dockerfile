FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04
ENV TZ="Europe/Tallinn"
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
   python3 \
   python3-pip \
   ffmpeg libsm6 libxext6 \
   git \
   wget \
   && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/supervisely-ecosystem/pips.git /pips_tracker

RUN pip3 install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install -r /pips_tracker/requirements.txt

RUN mkdir /pips_tracker/reference_model
RUN cd /pips_tracker/reference_model && wget --no-check-certificate 'https://github.com/supervisely-ecosystem/pips/releases/download/v0.0.1/model-000200000.pth' -O model-000200000.pth

RUN apt update && apt install -y python-is-python3
RUN mv /pips_tracker/ /root/pips_tracker/