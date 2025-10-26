import torch
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file
from datetime import datetime
import os

# --- MODEL AND PATH CONFIGURATION ---

# IMPORTANT: The path to your trained LoRA model file
# Assuming FriendFace_Lora is in your home directory (~) and the file is 'last.safetensors'
# If you are running this script from the FriendFace_Lora directory, the paths below are relative.
PROJECT_ROOT = os.path.expanduser("~/FriendFace_Lora")
LORA_PATH = os.path.join(PROJECT_ROOT, "model_512", "last.safetensors")

# Stable Diffusion 1.5 Base Model ID
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"

# --- GENERATION PARAMETERS ---

# The trigger word is "friendface person"
prompt = "photo of friendface person, professional headshot, high detail, studio lighting, corporate attire"

# Standard quality negative prompt for realistic images
negative_prompt = "bad anatomy, malformed, deformed, ugly, disfigured, blurry, low quality, duplicate, morbid, mutilated, extra fingers, worst quality, low resolution"

# Parameters you might want to adjust
num_inference_steps = 30  # Steps to run (20-30 is common)
guidance_scale = 7.5      # Strength of the prompt (7-8 is common)
lora_weight = 0.8         # Strength of the LoRA model (0.7 to 1.0 is common)
seed = 42                 # Set a fixed seed for reproducible results

# --- OUTPUT CONFIGURATION ---
# Output folder will be created inside the project root
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dynamic Filename Generation
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
prompt_slug = "_".join(prompt.split()[:4])
output_filename = f"{prompt_slug}_w{lora_weight}_g{guidance_scale}_{timestamp}.png"
output_path = os.path.join(OUTPUT_DIR, output_filename)

# --- EXECUTION ---

# Load the base model pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL_ID, 
    torch_dtype=torch.float16,
    safety_checker=None # Disable safety checker as requested
).to("cuda")

# Load the LoRA weights
try:
    if not os.path.exists(LORA_PATH):
        raise FileNotFoundError(f"LoRA model not found at: {LORA_PATH}")
        
    # Load the state dictionary from the .safetensors file
    state_dict = load_file(LORA_PATH)
    
    # Apply the LoRA weights to the pipeline
    pipe.load_lora_weights(state_dict, adapter_name="friendface")
    pipe.set_adapters("friendface", adapter_weight=lora_weight)

except FileNotFoundError as e:
    print(f"ERROR: {e}")
    print("Please ensure your trained 'last.safetensors' file is correctly placed in the model_512 folder.")
    exit()
except Exception as e:
    print(f"An error occurred during model loading: {e}")
    exit()

print(f"Generating image with prompt: {prompt}")
print(f"Output file: {output_path}")

# Generate the image
generator = torch.Generator("cuda").manual_seed(seed)
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
    generator=generator
).images[0]

# Save the image
image.save(output_path)
print(f"Image saved successfully to {output_path}")
