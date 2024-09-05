import ast
import base64
import os
from io import BytesIO
from pathlib import Path

import polars as pl
import requests
from dotenv import load_dotenv
from pdf2image import convert_from_path

def load_environment():
    load_dotenv()

def convert_pdf_to_images(file_path):
    return convert_from_path(file_path)

def encode_images_to_base64(images):
    image_b64 = []
    for image in images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_b64.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
        )
    return image_b64

def send_request_to_api(messages):
    api_key = os.getenv("OPENAI_API_KEY")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {"model": "gpt-4o-mini", "messages": messages}
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    return response.json()

def main():
    load_environment()

    prompt = """
    The following images are from a planning response form completed by a member of the public. They contain free-form responses related to a planning application, which may be either handwritten or typed.

    Please extract all the free-form information from these images and output it verbatim. Do not include any additional information or summaries. Note that the images are sequentially ordered, so a response might continue from one image to the next.
    """

    path = Path("./data/raw/pdfs")
    for file in path.glob("*.pdf"):
        if file.stem:
            images = convert_pdf_to_images(file)
            image_b64 = encode_images_to_base64(images)

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

            response = send_request_to_api(messages)
            print(response)
            break

if __name__ == "__main__":
    main()
