#!/bin/bash

trap 'on_exit' SIGINT

on_exit() {
    rm -rf figures_*
    rm -rf pdfs
    mkdir pdfs
    exit 0
}

sudo apt-get update
sudo apt-get install tesseract-ocr
echo "TESSERACT INSTALLED"
sudo apt install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"
apt-cache policy docker-ce
sudo apt install docker.io
echo "DOCKER INSTALLED"
sudo apt install python3.12-venv
python3 -m venv ragenv
echo "VIRTUAL ENVIRONMENT CREATED"
source ragenv/bin/activate
echo "RUNNING RAG"
sudo docker run -p 8501:8501 pranavrao25/ragimage:image &
wait $!