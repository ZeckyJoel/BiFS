import os
import json
import base64
import concurrent.futures
from openai import OpenAI
from IPython.display import display, Image as IPImage
import random
from tqdm import tqdm
import time


prompt_excluder = """
<|Instruction|>
1. Select tags that may introduce distraction or are unnecessary for answering the question based on the image.
2. Provide a clear and concise reason for each selected tag, explaining why it may distract from the question or why it is unnecessary for answering the question.
3. Output the result in JSON format, including only the tag and excluded_reason fields.

<|Requirements|>
1. Tag Exclusion:
  • Select ONLY from the provided list of object tags.
  • NEVER introduce new tags or modify existing ones.
  • Preserve the exact content and case of the tags.
2. Output Format:
  • Output must be in JSON format: [{{"tag": "...", "excluded_reason": "..."}}, ...].
  • Do not include json code block markers or any additional text.
3. Reasoning:
  • Reasons must be concise, logically clear, and consistent with the image content.

<|Example|>
Input:
Question: What kind of computer is near the woman in blue?
Tags: ['Person', 'Wii Remote', 'Monitor', 'Table', 'Wall']
  
Output:
[
  {{"tag": "Person", "excluded_reason": "The question focuses on the computer, not the person."}},
  {{"tag": "Wii Remote", "excluded_reason": "The Wii Remote is not relevant to identifying the type of computer."}},
  {{"tag": "Table", "excluded_reason": "The table is not necessary for determining the type of computer."}},
  {{"tag": "Wall", "excluded_reason": "The wall does not provide information about the computer type."}}
]

<|Input|>
Question: {question}
Tags: {tags}

<|Output|>
"""

def excluder_generate(prompt, dataset_path, split, dataset_type, image_path, concurrency, output_path):

    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    client = OpenAI(
        api_key="xxx",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    def process_sample(sample):
        try:
            question = sample.get("question", "")
            tags = sample.get("tags", [])
            image_id = str(sample.get("image_id", "")).strip()
            padded_image_id = image_id.zfill(12)

            if dataset_type == "okvqa":
                filename = f"COCO_{split}2014_{padded_image_id}.jpg"
            elif dataset_type == "aokvqa":
                filename = f"{padded_image_id}.jpg"
            else:
                raise ValueError(f"dataset_type: {dataset_type}")

            image_file = os.path.join(image_path, filename)

            if not os.path.exists(image_file):
                raise FileNotFoundError(f"Not Found: {image_file}")

            with open(image_file, "rb") as img_file:
                image_bytes = img_file.read()
            encoded_image = base64.b64encode(image_bytes).decode("utf-8")
            image_data_uri = f"data:image/jpeg;base64,{encoded_image}"

            full_prompt = prompt.format(question=question, tags=tags)

            retries_400 = 0
            retries_json = 0
            max_retries_400 = 3
            max_retries_json = 2

            while True:
                try:
                    response = client.chat.completions.create(
                        model="qwen-vl-max-1119",
                        messages=[
                            {"role": "system", "content": "You are a excluder agent. Your role is to select object tags from the provided list that may introduce distraction or are unnecessary for answering the question."},
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
                                        "text": full_prompt
                                    }
                                ]
                            }
                        ],
                        temperature=0,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        seed=42,
                        response_format={"type": "json_object"},
                    )

                    excluder_output = json.loads(response.choices[0].message.content)
                    break  

                except Exception as e:
                    if "Error code: 400" in str(e):
                        retries_400 += 1
                        print(f"400 Error, retrying {retries_400} 次...")
                        if retries_400 >= max_retries_400:
                            print("Maximum retry attempts reached, skipping this sample")
                            return {
                                **sample,
                                "excluder_output": None
                            }

                    elif isinstance(e, json.JSONDecodeError):
                        retries_json += 1
                        print(f"JSON parsing error, retrying {retries_json} time...")
                        if retries_json >= max_retries_json:
                            print("Maximum JSON parsing retry attempts reached, skipping this sample")
                            return {
                                **sample,
                                "excluder_output": None
                            }
                    elif "Error code: 429" in str(e):
                        print("Rate limit reached, retrying after 8 seconds...")
                        time.sleep(10)  
                        continue
                    else:
                        raise

            result_dict = {
                **sample,
                "excluder_output": excluder_output  
            }

            return result_dict

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(process_sample, sample) for sample in dataset]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing samples"):
            result = future.result()
            if result:
                results.append(result)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results have been saved to: {output_path}")

