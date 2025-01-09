import base64
import os
from io import BytesIO

import requests
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from tqdm import tqdm

from planning_ai.common.utils import Paths

load_dotenv()


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


def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file using PyPDF2."""
    try:
        reader = PdfReader(file_path, strict=True)
        text = []
        for page in reader.pages:
            text.append(page.extract_text() or "")
        return "\n".join(text).strip()
    except Exception as e:
        print(e)
        return None


def main():
    pdfs = (Paths.RAW / "pdfs").glob("*.pdf")
    with open("planning_ai/preprocessing/prompts/ocr.txt", "r") as f:
        ocr_prompt = f.read()

    for file in tqdm(pdfs):
        outfile = Paths.STAGING / "pdfs" / f"{file.stem}.txt"

        try:
            images = convert_from_path(file)
            image_b64 = encode_images_to_base64(images)

            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": ocr_prompt}] + image_b64,
                }
            ]

            response = send_request_to_api(messages)
            if not "choices" in response:
                continue
            out = response["choices"][0]["message"]["content"]
            if outfile.exists():
                continue
            with open(outfile, "w") as f:
                f.write(out)
        except:
            continue


if __name__ == "__main__":
    main()
