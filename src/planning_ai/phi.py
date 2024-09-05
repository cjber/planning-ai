from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

model_id = "microsoft/Phi-3.5-vision-instruct"

# Note: set _attn_implementation='eager' if you don't have flash_attn installed
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation="flash_attention_2",
)

# for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
processor = AutoProcessor.from_pretrained(
    model_id, trust_remote_code=True, num_crops=16
)

images = []
placeholder = ""
path = Path("./data/raw/pdfs")
i = 1
for file in path.glob("*.pdf"):
    pdf_images = convert_from_path(file)
    for image in pdf_images:
        images.append(image)
        placeholder += f"<|image_{i}|>\n"
        i += 1

messages = [
    {
        "role": "user",
        "content": """
<|image_1|>

This image is an extract from a planning response form filled out by a member of the public. The form may contain typed or handwritten responses, including potentially incomplete or unclear sections. The purpose is to extract all relevant structured information for further analysis. 

The form may include:
1. A questionnaire with structured questions and responses.
2. Handwritten notes under the section titled 'Your comments:'.

Please extract the information in the following format:
- For structured questions, use the format:
  {"<question>": "<response>"}
  
  Example:
  {"Do you support the planning proposal?": "Yes"}

- For handwritten comments under 'Your comments:', extract them verbatim. If a word is illegible or unclear, use the token <UNKNOWN>. Do not attempt to infer the meaning of unclear text or complete missing parts. 

  Example:
  {"Your comments:": "I believe this proposal is beneficial, although <UNKNOWN> might need further review."}

Guidelines:
- Maintain accuracy and structure; do not add any assumptions about the content.
- Ensure that any section, whether filled out or left blank, is noted appropriately.
- Prioritise accurate transcription of handwritten text, avoiding inference of ambiguous words.

Please follow these instructions precisely to ensure the extracted data is structured, clear, and as accurate as possible.
""",
    },
]

prompt = processor.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

inputs = processor(prompt, images[0], return_tensors="pt").to("cuda:0")

generation_args = {"max_new_tokens": 1000, "do_sample": False}

generate_ids = model.generate(
    **inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args
)

# remove input tokens
generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print(response)
