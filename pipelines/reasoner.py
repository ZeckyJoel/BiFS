import os
import json
import base64
import concurrent.futures
from openai import OpenAI
from IPython.display import display, Image as IPImage
import random
from tqdm import tqdm
import time
from collections import Counter

prompt_knowledge = """
<|Instruction|>
1. Analyze the question, provided key imformation, and the answer of the question to construct a chain of thought linking the evidence to the answer. 
2. Integrate the provided key information and relevant knowledge to build a concise chain of thought toward the answer.
3. Wrap your output inside the <|Because of|> and <|So the answer is|> markers.

<|Requirements|>
1. Content:
   - The output must not contradict the image.
   - Make full use of the provided key information.
   - External knowledge may be introduced when necessary to complete the chain of thought.
   - Conciseness and accuracy are imperative, every chain of thought must deliver only what's essential, with absolute precision.
2. Output Format: 
   - Begin your reasoning with `<|Because of|>`.  
   - Follow with your chain of thought.
   - Conclude with `<|So the answer is|>` and then the answer.
   - Preserve the exact content and case of the answer.
   - Output text content only, without any additional tags or formatting.

<|Example|>
Input:
Question: What is placed inside this meter?
Key Imformation: [
      {{
        "tag": "Parking Meter",
        "attributes": {{
          "color": "red and gray",
          "text": "Denver's Road Home",
          "label": "Campaign to End Homelessness",
          "coin slot": "present",
          "instructions": "accepts coins"
        }},
        "caption": "The Parking Meter is stationary, placed on a sidewalk surrounded by greenery and a concrete wall. It serves as a donation station for ending homelessness.",
        "selected_reason": "The image shows a parking meter, which is the primary object in question. Understanding what is placed inside it requires identifying the meter itself."
      }}
    ]
Answer: coins

Output:<|Because of|> The parking meter is labeled "Campaign to End Homelessness" and has a coin slot, indicating it accepts coins as donations. <|So the answer is|> coins.

<|Input|>
Question: {question}
Key Imformation: {key_information}
Answer: {answer}

<|Output|>
"""

def most_frequent_string(string_list):
    counter = Counter(string_list)
    most_common = counter.most_common(1)
    return most_common[0][0] if most_common else None

def knowledge_generate(prompt_knowledge, dataset_path, split, dataset_type, image_path, concurrency, output_path, task_type):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    client = OpenAI(
        api_key="xxx",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    def process_sample(sample):
        try:
            question = sample.get("question", "")
            key_information = sample.get("key_information", [])
            direct_answers = sample.get("direct_answers", [])

            if dataset_type == "okvqa":
                answer = most_frequent_string(direct_answers)
            elif dataset_type == "aokvqa":
                if task_type == "MC":
                    choices = sample.get("choices", None)
                    correct_choice_idx = sample.get("correct_choice_idx", None)
                    answer = choices[correct_choice_idx] if correct_choice_idx and choices else None
                elif task_type == "DA":
                    answer = most_frequent_string(direct_answers)
                else:
                    raise ValueError(f"task_type: {task_type}")
            else:
                raise ValueError(f"dataset_type: {dataset_type}")

            if not answer:
                if dataset_type == "okvqa":
                    return {**sample, "knowledge": None}
                else:
                    return {**sample, "DA_knowledge": None, "MC_knowledge": None}

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

            full_prompt = prompt_knowledge.format(question=question, key_information=key_information, answer=answer)

            retries = 0
            max_retries = 6
            knowledge_output = None

            while retries < max_retries:
                try:
                    response = client.chat.completions.create(
                        model="qwen-vl-max-latest",
                        messages=[
                            {"role": "system", "content": "You are a professional reasoner. Your role is to generate a concise chain of thought toward the answer, based on the provided image and question, combined with key information and the given answer."},
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
                        seed=42
                    )

                    raw_response = response.choices[0].message.content.strip()

                    if "<|Because of|>" in raw_response and "<|So the answer is|>" in raw_response:
                        start_idx = raw_response.find("<|Because of|>") + len("<|Because of|>")
                        end_idx = raw_response.find("<|So the answer is|>")
                        knowledge = raw_response[start_idx:end_idx].strip()
                        knowledge_output = knowledge
                        break
                    else:
                        retries += 1
                except Exception as e:
                    if "Error code: 429" in str(e):
                        print("Rate limit reached, retrying after 8 seconds...")
                        time.sleep(8)  
                        continue
                    else:
                        retries += 1

            if knowledge_output is None:
                if dataset_type == "okvqa":
                    return {
                        **sample,
                        "knowledge": None
                    }
                else:
                    return {
                        **sample,
                        "DA_knowledge": None,
                        "MC_knowledge": None
                    }

            if dataset_type == "okvqa":
                result = {
                    **sample,
                    "knowledge": knowledge_output
                }
            else:
                result = {
                    **sample,
                    "DA_knowledge": knowledge_output if task_type == "DA" else sample.get("DA_knowledge"),
                    "MC_knowledge": knowledge_output if task_type == "MC" else sample.get("MC_knowledge")
                }

            return result

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