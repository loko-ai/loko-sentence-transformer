FROM node:16.15.0 AS builder
ADD ./frontend/package.json /frontend/package.json
WORKDIR /frontend
RUN yarn install
ADD ./frontend /frontend
RUN yarn build --base="/routes/sentence_transformer/web/"

FROM nvidia/cuda:11.8.0-base-ubuntu22.04
RUN apt update && apt upgrade -y && apt install -y python3.10 python3.10-dev python3-pip
RUN update-alternatives --install /usr/local/bin/python python /usr/bin/python3.10 0
RUN python -m pip install --upgrade pip
EXPOSE 8080
ADD ./requirements.txt /
RUN pip install -r /requirements.txt
ARG GATEWAY
ENV GATEWAY=$GATEWAY
ADD . /plugin
ENV PYTHONPATH=$PYTHONPATH:/plugin
WORKDIR /plugin/services
CMD python services.py