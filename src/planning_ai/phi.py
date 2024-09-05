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

This image is an extract from a planning response form filled out by a member of the public. The form may contain typed or handwritten responses, including potentially incomplete or unclear sections. Your task is to extract relevant information in a strict, structured format. Do not repeat the document verbatim. Only output responses in the structured format below.

Instructions:
1. Extract responses to all structured questions on the form, in the format:
   {"<question>": "<response>"}
   
   Example:
   {"Do you support the planning proposal?": "Yes"}

2. For the handwritten notes under 'Your comments:', extract them verbatim. If any word is illegible or unclear, use the token <UNKNOWN>. Do not attempt to infer or complete missing parts. Use the format:
   {"Your comments:": "<verbatim comments>"}
   
   Example:
   {"Your comments:": "I support the proposal, but the <UNKNOWN> aspect requires attention."}

3. **Do not** output or repeat the original document content in full. Only return structured data in the format described above.
4. **Ignore irrelevant sections** that are not part of the structured questionnaire or 'Your comments:' section.
5. If a response is missing or the form section is blank, output:
   {"<question>": "No response"}

Guidelines:
- Ensure you return only structured data in JSON-like format.
- Strictly follow the format for both structured questions and handwritten comments.
- If any part of the form is unclear or unreadable, do not fill it in with assumptions.
- Avoid repeating the full content of the form. Focus only on extracting the relevant sections.

Example output:
{
  "Do you support the planning proposal?": "Yes",
  "Your comments:": "The proposal seems reasonable, but <UNKNOWN> needs further assessment."
}
""",
    }
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
