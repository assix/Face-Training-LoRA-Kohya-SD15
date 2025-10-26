FriendFace LoRA Training Project (Stable Diffusion v1.5)

This repository contains the scripts and configuration necessary to train a highly effective LoRA (Low-Rank Adaptation) model on the Stable Diffusion v1.5 base model. The goal is to embed a specific identity, triggered by the keyword friendface person, into the model.

🚀 Quick Start (Training)

These instructions assume you have cloned the kohya_ss repository and placed it directly in your home directory (~/kohya_ss).

1. Prepare Data & Directories

Before training, your image data must be organized in the following strict structure, which will be referenced by the training script:

~/FriendFace_Lora/
├── images_512/
│   └── 10_friendface_person/
│       ├── friend-image-1.png
│       ├── friend-image-1.txt
│       ├── friend-image-2.png
│       ├── friend-image-2.txt
│       └── ...
├── model_512/  (Will be created by the script)
├── log_512/    (Will be created by the script)
├── train_sd15_lora.sh
└── generate_sd15.py


Image Format: All images must be 512x512 pixels.

Renaming: Rename the directory containing your 512x512 images to 10_friendface_person. The 10 represents the number of repeats (recommended starting point) and friendface person is your trigger keyword.

Captioning: Each image (.png) requires an accompanying text file (.txt) with the same name, containing a detailed description of the image content without mentioning the trigger phrase.

2. Run Training

The included train_sd15_lora.sh script handles environment activation and launches the training process for 1200 steps.

Grant Execution Permissions:

chmod +x train_sd15_lora.sh


Execute the Training:

./train_sd15_lora.sh


Note: The script relies on the accelerate launch utility being configured and the kohya_ss dependencies being installed (including the correct version of PyTorch for your GPU architecture).

✨ Generation (Inference)

Once training is complete, the generate_sd15.py script can be used to test your new LoRA model. It will read the parameters from the script and save the output image to a newly created ./output/ directory.

Ensure Virtual Environment is Active (if not already):

source ~/kohya_ss/venv/bin/activate


Run the Generation Script:

python generate_sd15.py


Key Generation Parameters

The primary trigger is friendface person. Use it at the beginning of your positive prompts to activate the LoRA model.

Parameter

Recommended Range

Description

prompt

Varies

Must include friendface person.

lora_weight

0.7 to 1.0

Controls the strength of the model's influence.

guidance_scale

7.0 to 8.0

Controls how strictly the image follows the prompt.

num_inference_steps

20 to 30

Number of steps to generate the image.

negative_prompt

N/A

Ensures high-quality images and avoids defects.