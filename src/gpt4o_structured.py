from io import BytesIO
from pathlib import Path

from pdf2image import convert_from_path

prompt = """
This image is an extract from a planning response form filled out by a member of the public. The form may contain typed or handwritten responses, including potentially incomplete or unclear sections. Your task is to extract relevant information in a strict, structured format. Do not repeat the document verbatim. Only output responses in the structured format below.

Instructions:
1. Extract responses to all structured questions on the form, in the format:
   {"<question>": "<response>"}
   
2. For the handwritten notes under extract them verbatim. If any word is illegible or unclear, use the token <UNKNOWN>. Do not attempt to infer or complete missing parts.
   
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
"""

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

import base64

buffered = BytesIO()
images[2].save(buffered, format="JPEG")
base64_image = base64.b64encode(buffered.getvalue())

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt,
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
        ],
    }
]
import requests

api_key = "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"
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
payload = {"model": "gpt-4o", "messages": messages}


response = requests.post(
    "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
)

print(response.json()["choices"][0]["message"])
