import ast
import base64
import os
from io import BytesIO
from pathlib import Path

import polars as pl
import requests
from dotenv import load_dotenv
from pdf2image import convert_from_path

load_dotenv()

prompt = """
The following images are from a planning response form completed by a member of the public. They contain free-form responses related to a planning application, which may be either handwritten or typed.

Please extract all the free-form information from these images and output it verbatim. Do not include any additional information or summaries. Note that the images are sequentially ordered, so a response might continue from one image to the next.
"""

placeholder = ""
path = Path("./data/raw/pdfs")
i = 1
for file in path.glob("*.pdf"):
    images = []
    if file.stem:
        pdf_images = convert_from_path(file)
        for image in pdf_images:
            images.append(image)
            placeholder += f"<|image_{i}|>\n"
            i += 1

    buffered = BytesIO()
    outs = []
    image_b64 = []
    for image in images:
        image.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        image_b64.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
        )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
            ]
            + image_b64,
        }
    ]

    api_key = os.getenv("OPENAI_API_KEY")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {"model": "gpt-4o-mini", "messages": messages}

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    response.json()
    break
