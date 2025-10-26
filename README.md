Face-Training-LoRA-Kohya-SD15

This repository contains the necessary scripts and configuration to train a customized LoRA (Low-Rank Adaptation) model using the Stable Diffusion v1.5 base model and the kohya_ss training framework. The goal is to embed a specific personal identity, triggered by the keyword friendface person, into the diffusion model for creative image generation.

⚙️ Prerequisites

This project assumes the following directory structure, with all directories located directly under your home directory (~).

~/
├── kohya_ss/            # Complete and installed kohya_ss repository
│   └── venv/            # Python Virtual Environment
├── FriendFace_Lora/     # THIS REPOSITORY (Project Root)
│   ├── images_512/      # Parent directory for training data
│   ├── model_512/       # Output folder for trained LoRA files
│   └── log_512/         # Training logs and session data


Environment Setup

Clone and Setup Kohya: Ensure you have the kohya_ss repository cloned and the dependencies installed.

git clone [https://github.com/bmaltais/kohya_ss.git](https://github.com/bmaltais/kohya_ss.git) ~/kohya_ss
cd ~/kohya_ss
./setup.sh


Activate Virtual Environment: Activate the Python environment before running any scripts.

source ~/kohya_ss/venv/bin/activate


🚀 Training (SD v1.5)

The included script handles environment activation and launches the training process for 1200 steps at the native 512x512 resolution.

1. Prepare Data & Directories

Your image data must be placed inside the images_512/ directory in a subdirectory following a strict naming convention: <repeats>_<trigger_keyword>.

Directory Path: ~/FriendFace_Lora/images_512/10_friendface_person/

Filename Element

Example

Description

Repeats

10

How many times each image is repeated per epoch.

Trigger Keyword

friendface person

The custom phrase used to activate your LoRA during generation.

Files Required:

Images: All must be 512x512 pixels (.png is preferred).

Captions: Each image (.png) must have an accompanying text file (.txt) with the identical filename, containing a detailed description of the image without the trigger phrase (friendface person).

2. Run Training

The script train_sd15_lora.sh is configured with all the required parameters.

Grant Execution Permissions:

cd ~/FriendFace_Lora
chmod +x train_sd15_lora.sh


Execute the Training:

./train_sd15_lora.sh


✨ Generation (Inference)

Once training is complete, the generate_sd15.py script can be used to test your new LoRA model. It will load the model from ~/FriendFace_Lora/model_512/.

Run the Generation Script:

cd ~/FriendFace_Lora
python generate_sd15.py


Key Generation Parameters

The primary trigger is friendface person. Use it at the beginning of your positive prompts to activate the LoRA model.

Parameter

Default Value

Usage

prompt

Varies

Must include friendface person.

lora_path

last.safetensors

Points to the trained model file.

lora_weight

0.85

Controls the strength of the model's influence (recommended: 0.7 to 1.0).

guidance_scale

7.5

Controls how strictly the image follows the prompt (recommended: 7.0 to 8.0).

negative_prompt

(High quality)

Ensures high-quality images and avoids defects.