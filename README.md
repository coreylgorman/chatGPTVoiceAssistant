# chatGPTVoiceAssistant

# Originally developed for Pearl STEM Academy 2023

# chatGPTWhisperAPI4JNano
Utilizes openAI API for Whisper and chatGPT targeting a Jetson Nano

# Assistant on Jetson Nano Developer Kit

This repository contains the code and setup instructions to run a voice-activated assistant on a Jetson Nano Developer Kit.

## Prerequisites

- NVIDIA Jetson Nano Developer Kit
- Microphone connected to the Jetson Nano
- JetPack installed on the Jetson Nano

## Installation

1. Install JetPack on the Jetson Nano Developer Kit:
   - From a separate computer download the image to an sd card:
   -    https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit

X. Download .zip file from github repo and extract to the following location (remove "-main" from .zip file name before extracting)
   - user\Projects\STEM2023

﻿
# Basic system update and install of dependencies for Whisper and chatGPT on Jetson Nano
# Installation instructions "installationInstructionsLatest.txt" may be easier to read and has correct formatting

sudo apt-get update
sudo apt install nano

echo "Installing dependencies for pyenv and Python build..."
sudo apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev python-openssl git

echo "Installing Pyenv..."
curl https://pyenv.run | bash

# Add init lines to bashrc, they will be used every time a new shell is started
echo "Adding Pyenv to bashrc..."
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc


echo "Applying changes to the current shell..."
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"


echo "Installing Python 3.9.16 using Pyenv and setting it as default..."
pyenv install 3.9.16

echo "Creating a new virtual environment using Pyenv..."
pyenv virtualenv 3.9.16 myenv

#change directory to Project folder
cd Projects/STEM2023/chatGPTWhisperAPI4JNano-main

echo "Activating the virtual environment..."
pyenv activate myenv

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing pyaudio dependency..."
sudo apt-get install -y portaudio19-dev

echo "Installing Python packages..."
pip install -r requirements.txt

update openai api key in secretkeys.py 


Try running the project
python3 assistant.py

# Create a new file called assistant.service
sudo nano /etc/systemd/system/assistant.service

# In the file assistant.service that opens, include the following lines
[Unit]
Description=My Voice Assistant
After=network.target

[Service]
ExecStart=/home/pearl/.pyenv/versions/myenv/bin/python /home/pearl/Projects/STEM2023/chatGPTVoiceAssistant/assistant.py
WorkingDirectory=/home/pearl/Projects/STEM2023/chatGPTVoiceAssistant/
StandardOutput=inherit
StandardError=inherit
Restart=always
User=pearl

[Install]
WantedBy=multi-user.target" | sudo tee /etc/systemd/system/assistant.service

# Save and Exit


sudo systemctl enable assistant.service
sudo systemctl start assistant.service


# Save the file and Exit (ctl+x, ctl+y, enter)

# Set the permissions for the service file:
sudo chmod 644 /etc/systemd/system/assistant.service


echo "Reloading systemd manager configuration..."
sudo systemctl daemon-reload
sudo systemctl restart assistant.service

echo "Enabling the service to start on boot..."
sudo systemctl enable assistant.service

echo "Starting the service..."
sudo systemctl start assistant.service

echo "All done!"




Here are some useful commands for managing and troubleshooting services in Linux:

systemctl start service_name: Start a service.
systemctl stop service_name: Stop a service.
systemctl restart service_name: Restart a service.
systemctl reload service_name: Reload the configuration of a service without restarting it.
systemctl status service_name: Check the status of a service.
systemctl enable service_name: Enable a service to start automatically on system boot.
systemctl disable service_name: Disable a service from starting automatically on system boot.
systemctl is-active service_name: Check if a service is currently active.
systemctl is-enabled service_name: Check if a service is enabled to start on system boot.
systemctl list-unit-files: List all installed unit files (services).
journalctl -u service_name: View the logs of a service.
systemctl reload-daemon: Reload the systemd daemon to apply changes in unit files.
Replace service_name with the actual name of the service you want to manage or troubleshoot.

These commands should help you effectively manage and troubleshoot services on your system.
