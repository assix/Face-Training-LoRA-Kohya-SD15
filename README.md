# **Face-Training-LoRA-Kohya-SD15**

This repository provides scripts and configuration files for training a custom **LoRA (Low-Rank Adaptation)** model using the **Stable Diffusion v1.5** base model with the **kohya_ss** training framework.
The purpose is to embed a **specific personal identity**, activated by the trigger keyword **`friendface person`**, into the diffusion model for use in creative image generation.

---

## ⚙️ **Prerequisites**

This project assumes the following directory structure, with each folder located directly under your home directory (`~/`):

```
~/
├── kohya_ss/                 # Full kohya_ss installation
│   └── venv/                 # Python virtual environment
├── Face-Training-LoRA-Kohya-SD15/     # This repository (project root)
│   ├── images_512/           # Training data
│   ├── model_512/            # Output LoRA models
│   └── log_512/              # Training logs and sessions
```

---

## 🛠️ **Environment Setup**

### **1. Clone and Set Up Kohya**

```bash
git clone https://github.com/bmaltais/kohya_ss.git ~/kohya_ss
cd ~/kohya_ss
./setup.sh
```

### **2. Activate the Virtual Environment**

```bash
source ~/kohya_ss/venv/bin/activate
```

---

## 🚀 **Training (Stable Diffusion v1.5)**

The repository includes a full training script configured for **1200 steps** at **512×512 resolution**.

---

## **1. Prepare Your Data & Directories**

Your training images must be placed inside a folder named using the format:

```
<repeats>_<trigger_keyword>
```

Example directory:

```
~/Face-Training-LoRA-Kohya-SD15/images_512/10_friendface_person/
```

### **Naming Breakdown**

| Element             | Example             | Description                                       |
| ------------------- | ------------------- | ------------------------------------------------- |
| **Repeats**         | `10`                | Number of times each image is repeated per epoch. |
| **Trigger Keyword** | `friendface person` | Phrase used to activate the LoRA in prompts.      |

---

### 📸 **Image Requirements (IMPORTANT)**

All images in `images_512/` must meet the following:

* **Resolution:** Exactly **512×512 px**
* **Quantity:** **15–30** high-quality images recommended
* **Variety:** Different lighting, expressions, angles, and outfits
* **Backgrounds:** Avoid too much repetition or overly complex scenery
* **Caption Files:**

  * Each `.png` must have a **same-name `.txt` caption**
  * Descriptions **must not** include the trigger phrase

#### **Example File Pair**

| File                    | Content                                                                     | Purpose                                |
| ----------------------- | --------------------------------------------------------------------------- | -------------------------------------- |
| `friendface-image1.png` | (512×512 image)                                                             | Source photo                           |
| `friendface-image1.txt` | `a woman wearing a red sweater, sitting on a wooden bench, park background` | Image description (no trigger keyword) |

---

## **2. Run Training**

### **Give Execution Permission**

```bash
cd ~/Face-Training-LoRA-Kohya-SD15
chmod +x train_sd15_lora.sh
```

### **Start Training**

```bash
./train_sd15_lora.sh
```

Training output will be saved in:

* **model_512/** — trained LoRA models
* **log_512/** — training logs

---

## ✨ **Generation (Inference)**

Once training is complete, run the generation script to test your custom LoRA model.

### **Run the Script**

```bash
cd ~/Face-Training-LoRA-Kohya-SD15
python generate_sd15.py
```

---

## 🎨 **Key Generation Parameters**

The **primary trigger phrase** is:

```
friendface person
```

Use it at the beginning of your positive prompt.

| Parameter           | Default Value         | Description                                       |
| ------------------- | --------------------- | ------------------------------------------------- |
| **prompt**          | varies                | Must include `friendface person`.                 |
| **lora_path**       | `last.safetensors`    | Path to your trained LoRA.                        |
| **lora_weight**     | `0.85`                | Model influence (recommended: 0.7–1.0).           |
| **guidance_scale**  | `7.5`                 | Prompt adherence strength (recommended: 7.0–8.0). |
| **negative_prompt** | high-quality defaults | Helps avoid defects and artifacts.                |


