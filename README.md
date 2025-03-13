# tagRating
Llama-3.2-11B-Vision-Instruct-TagRater The Llama-3.2-11B-Vision-Instruct-TagRater is a merged multi-modal model designed to rate images based on a provided tagword. By combining visual and language understanding, this model evaluates an image against a rating prompt and produces a concise explanation along with a relevance rating from 0 to 5.

# Pixu.ai Vision-Instruct TagRater

An advanced image relevance evaluator that determines how well an image matches a given search term. Built upon state-of-the-art vision and language models, this project leverages efficient fine-tuning and quantization techniques to deliver both performance and precision.

---

## Table of Contents

- [Overview](#overview)
- [Model Architecture & Training](#model-architecture--training)
- [Rating Guidelines](#rating-guidelines)
- [Performance Metrics](#performance-metrics)
- [Usage Example](#usage-example)
- [Installation](#installation)
- [Credits](#credits)

---

## Overview

This project is based on the unsloth/Llama-3.2-11B-Vision-Instruct model and has been re-engineered for optimal image and text processing. With a fully merged model deployed in a memory-efficient 4-bit mode, it is ready to use immediately, making it ideal for rapid integration into your applications.

---

## Model Architecture & Training

- **Base Model:** unsloth/Llama-3.2-11B-Vision-Instruct
- **Architecture:** FastVisionModel, designed for efficient visual processing.
- **Fine-Tuning:**  
  - Applied LoRA fine-tuning on vision, language, attention, and MLP layers.
  - Employed a supervised fine-tuning strategy using a conversational data format that integrates both image and text inputs.
  - Techniques like gradient checkpointing and 8-bit AdamW were used to enhance training efficiency.
- **Quantization:** Loaded in 4-bit mode for reduced memory footprint.
- **Deployment:** Fully merged and ready for immediate use without additional assembly steps.

**Training Insights:**

- **Data Preparation:**  
  - Images were resized to 512×512 pixels.
  - Each sample pairs an image with a prompt instructing the model to evaluate the image against a specific search term.
- **Fine-Tuning Configuration:**  
  - Utilizes LoRA parameters (e.g., r=16, lora_alpha=16) to optimize both visual and textual components.

---

## Rating Guidelines

The model was trained using a detailed instruction set to assess image relevance. The prompt provided during training was:

> **"Evaluate how well this image matches the search term: [tagword]. Provide a concise explanation and assign a score between 0 and 5."**

**Scoring System:**

- **0 – Not Relevant:** No connection observed.
- **1 – Barely Relevant:** Very weak or vague link.
- **2 – Minimally Relevant:** Slight hints, but lacks clarity.
- **3 – Moderately Relevant:** Noticeable connection, though not dominant.
- **4 – Highly Relevant:** Strong and clear representation.
- **5 – Perfectly Relevant:** Exemplary match with the search term.

**Evaluation Factors:**

- **Content Relevance:** Clarity of the relationship between image and search term.
- **Context & Setting:** Suitability of the image’s style and theme.
- **Visual Appeal & User Satisfaction:** Overall user engagement and satisfaction with the image.

---

## Performance Metrics

During inference, the model achieves:

- **Tokens Generated:** Approximately 233 per session.
- **Processing Speed:** About 40.65 tokens per second.
- **Memory Usage:**  
  - **VRAM:** ~7835 MB allocated and ~8862 MB reserved.
  - **RAM:** ~2281 MB in use.

These metrics showcase the model’s computational efficiency and resource management.

---

## Usage Example

Below is a Python script example demonstrating how to initialize and run the model:

```python
from unsloth import FastVisionModel
import base64, json, re
from io import BytesIO
from PIL import Image
import torch

# --- Model Initialization ---
model, tokenizer = FastVisionModel.from_pretrained(
    model_name="Pixuai/Llama-3.2-11B-Vision-Instruct-TagRater",
    load_in_4bit=True,
    max_seq_length=150,
)
FastVisionModel.for_inference(model)

# --- Prepare Input ---
prompt = (
    "Evaluate how well this image matches the search term: space explorer. "
    "Provide a concise explanation and assign a score (0-5) based on the criteria:\n"
    "0 – Not Relevant: No connection.\n"
    "1 – Barely Relevant: Very weak or vague link.\n"
    "2 – Minimally Relevant: Slight hints but lacks clarity.\n"
    "3 – Moderately Relevant: Noticeable connection but not dominant.\n"
    "4 – Highly Relevant: Strong and clear representation.\n"
    "5 – Perfectly Relevant: Exemplary match."
)
image_b64 = "base64_string_here"  # Replace with a valid base64 encoded image string

try:
    image_data = base64.b64decode(image_b64)
    image = Image.open(BytesIO(image_data)).convert("RGB")
except Exception as e:
    print(f"Error decoding image: {str(e)}")
    exit()

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ],
    }
]

# --- Tokenize Inputs ---
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")

# --- Model Inference ---
gen_tokens = model.generate(
    **inputs,
    max_new_tokens=100,
    use_cache=True,
    temperature=0.1,
    min_p=0.1,
)

# --- Output Decoding ---
output_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
json_match = re.search(r'({.*})', output_text, re.DOTALL)
if json_match:
    json_str = json_match.group(1)
    try:
        json_obj = json.loads(json_str)
    except json.JSONDecodeError:
        print("Invalid JSON output.")
        exit()
    print("JSON Output:", json_obj)
else:
    print("No JSON object found in the output.")
