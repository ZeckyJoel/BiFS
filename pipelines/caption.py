import os
import json
import base64
import concurrent.futures
from openai import OpenAI
from IPython.display import display, Image as IPImage
import random
from tqdm import tqdm

prompt_cap = """
<|Instruction|>
Please perform an image description task: identify the objects in the image along with their attributes, and provide a brief description of each object.

<|Requirements|>
Create an entry for each object in the image.
Each entry should include:
  • tag: A concise name for the object.
  • attributes: The object's physical and visual attributes, including any text or symbols present on it, which are not influenced by the surrounding environment.
  • caption: A descriptive sentence using the object as the subject, which should include:
    1. What is the object currently experiencing in the environment (e.g., is it being operated, or in motion)?
    2. What connection, interaction, or relationship does it have with other objects or the environment in the image (e.g., "resting on X", "surrounded by Y", "placed next to Z")?
    3. What role or function does it serve in this environment (e.g., "serves as decoration", "provides lighting")?

<|Output|>
Output in JSON format: `[{"tag": "...", "attributes": {"key1": "...", "key2": "...", ... }, "caption": "..."}, ...]`. Do not include ```json``` code block markers or any additional text.
"""

def generate_image_descriptions(prompt, dataset_path, split, dataset_type, image_path, concurrency, output_path):

    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    client = OpenAI(
        api_key="xxx",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    def process_sample(sample):
        try:
            question_id = sample.get("question_id", None)
            image_id = str(sample["image_id"]).zfill(12)

            if dataset_type == "okvqa":
                image_filename = f"COCO_{split}2014_{image_id}.jpg"
            elif dataset_type == "aokvqa":
                image_filename = f"{image_id}.jpg"
            else:
                raise ValueError("Unsupported dataset type")

            image_file = os.path.join(image_path, image_filename)

            if not os.path.exists(image_file):
                raise FileNotFoundError(f"Not Found: {image_file}")

            with open(image_file, "rb") as img_file:
                image_bytes = img_file.read()
            encoded_image = base64.b64encode(image_bytes).decode("utf-8")
            image_data_uri = f"data:image/jpeg;base64,{encoded_image}"

            retries_400 = 0
            retries_json = 0
            max_retries_400 = 3
            max_retries_json = 2

            while True:
                try:
                    response = client.chat.completions.create(
                        model="qwen-vl-max",
                        messages=[
                            {"role": "system", "content": [{"type": "text", "text": "You are a professional visual description assistant."}]},
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": image_data_uri
                                        }
                                    },
                                    {
                                        "type": "text",
                                        "text": prompt
                                    }
                                ]
                            }
                        ],
                        response_format={"type": "json_object"}
                    )

                    image_caption = json.loads(response.choices[0].message.content)
                    break  

                except Exception as e:
                    if "Error code: 400" in str(e):
                        retries_400 += 1
                        print(f"400 Error, retrying  {retries_400} time...")
                        if retries_400 >= max_retries_400:
                            print("Maximum retry attempts reached, skipping this sample")
                            return {
                                **sample,
                                "image_caption": None
                            }

                    elif isinstance(e, json.JSONDecodeError):
                        retries_json += 1
                        print(f"JSON parsing error, retrying {retries_json} time...")
                        if retries_json >= max_retries_json:
                            print("Maximum JSON parsing retry attempts reached, skipping this sample")
                            return {
                                **sample,
                                "image_caption": None
                            }
                    else:
                        raise

            result_dict = {
                **sample,
                "image_caption": image_caption
            }

            return result_dict

        except Exception as e:
            print("Rate limit reached, retrying after 8 seconds...")
            return None

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(process_sample, sample) for sample in data]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing samples"):
            result = future.result()
            if result:
                results.append(result)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results have been saved to: {output_path}")