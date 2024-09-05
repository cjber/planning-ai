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
<|image_1|>\nThis image shows an extract from a planning response form filled out by a member of the public. They may be pro or against the planning proposal. These planning applications typically cover the construction of new buildings, or similar infrastructure.

Extract all structured information from these documents. For example a section may include a questionnaire that may or may not have been filled in. Please indicate the response from the member of public in a structured format, following the convention:

{"<question>": "<response>"}

The document may also include hand written notes under the title 'Your comments:', also include these notes verbatim, in a structured format. If a word is unreadable please use the special token <UNKNOWN>. Do not attempt to fill in the word if you are unsure.
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
