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


prompt_judge = """
<|Instruction|>
1. For each object in the Conflict Tags, evaluate the select_reason and exclude_reason based on the provided image , question, and corresponding answer.
2. The `select_reason` explains why the object is considered key information that is essential for answering the question, and `exclude_reason` explains why it is considered noisy information that may introduce distraction or is unnecessary for answering the question.  
2. Score each reason according to the five criteria:  
   • QR (Question Relevance): evaluates the reason's relevance to the main subject of the question.  
   • IC (Image Consistency): checks whether the reason accurately reflects the content of the image.  
   • RC (Reasoning Clarity): assesses whether the reasoning is logically coherent and easy to follow.  
   • IU (Information Utility): measures how much the reason provides useful clues toward deriving the answer.  
   • NA (Noise Avoidance): gauges the extent to which the reason avoids introducing irrelevant or distracting details.  
3. Scores should be integers between 1 and 5, where 1 indicates minimal adherence to the criterion and 5 indicates full adherence to the criterion.  
4. Output the result in JSON format, including the `tag`, `select_score` and `exclude_score` fields.

<|Requirements|>
1. Tag Selection:
   • Select ONLY from the provided list of conflict tags.
   • NEVER introduce new tags or modify existing ones.
   • Preserve the exact content and case of the tags.
2. Output Format:
   • Output must be in JSON format: [
      {{  
        "tag": "...",  
        "select_score": {{ "QR":int, "IC":int, "RC":int, "IU":int, "NA":int }},  
        "exclude_score": {{ "QR":int, "IC":int, "RC":int, "IU":int, "NA":int }}  
      }},
      ...
    ]  
   • Do not include json code block markers or any additional text.

<|Example|>
Input:
Question: If this is a grand slam which must it be?
Answer: wimbledon
Conflict Tags: [
  {{
    "tag": "Tennis Player",
    "select_reason": "The presence of a tennis player is essential as it confirms the activity is a tennis match.",
    "exclude_reason": "The question is about the event, not the player."
  }},
  {{
    "tag": "Net",
    "select_reason": "The net is part of the court setup and helps confirm the context of a tennis match.",
    "exclude_reason": "The net is a standard part of a tennis court and does not indicate the event."
  }},
  {{
    "tag": "Tennis Court",
    "select_reason": "The type of court (grass) is a key indicator of which Grand Slam tournament it might be, as Wimbledon is the only Grand Slam played on grass.",
    "exclude_reason": "The court type is not specific to identifying the grand slam event."
  }}
]

Output: [
  {{
    "tag": "Tennis Player",
    "select_score": {{ "QR": 4, "IC": 5, "RC": 5, "IU": 3, "NA": 4 }}, 
    "exclude_score": {{ "QR": 2, "IC": 5, "RC": 5, "IU": 1, "NA": 5 }} 
  }},
  {{
    "tag": "Net",
    "select_score": {{ "QR": 2, "IC": 5, "RC": 5, "IU": 2, "NA": 3 }}, 
    "exclude_score": {{ "QR": 4, "IC": 5, "RC": 5, "IU": 4, "NA": 5 }} 
  }},
  {{
    "tag": "Tennis Court",
    "select_score": {{ "QR": 5, "IC": 5, "RC": 5, "IU": 5, "NA": 5 }},
    "exclude_score": {{ "QR": 2, "IC": 5, "RC": 5, "IU": 1, "NA": 5 }} 
  }}
]

<|Input|>
Question: {question}
Answer: {answer}
Conflict Tags: {conflict_tags}

<|Output|>
"""


def most_frequent_string(string_list):
    counter = Counter(string_list)
    most_common = counter.most_common(1)
    return most_common[0][0] if most_common else None


def judge_generate_scores(prompt_judge, dataset_path, split, dataset_type, image_path, concurrency, output_path):

    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    client = OpenAI(
        api_key="xxx",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    def process_sample(sample):
        try:
            question = sample.get("question", "")
            conflict_tags = sample.get("conflict_tags", None)
            direct_answers = sample.get("direct_answers", [])

            if conflict_tags is None:
                sample["judge_output"] = None
                return sample
            
            answer = most_frequent_string(direct_answers) 

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

            full_prompt = prompt_judge.format(question=question, answer=answer, conflict_tags=conflict_tags)

            retries_400 = 0
            retries_json = 0
            max_retries_400 = 3
            max_retries_json = 2

            while True:
                try:
                    response = client.chat.completions.create(
                        model="qwen-vl-max-latest",
                        messages=[
                            {"role": "system", "content": "As a professional judge, your role is to evaluate and score the provided reason based on the given information and established standards."},
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

                    judge_output = json.loads(response.choices[0].message.content)
                    break  

                except Exception as e:
                    if "Error code: 400" in str(e):
                        retries_400 += 1
                        print(f"400 Error, retrying  {retries_400} time...")
                        if retries_400 >= max_retries_400:
                            print("Maximum retry attempts reached, skipping this sample")
                            return {
                                **sample,
                                "judge_output": None
                            }

                    elif isinstance(e, json.JSONDecodeError):
                        retries_json += 1
                        print(f"JSON parsing error, retrying {retries_json} time...")
                        if retries_json >= max_retries_json:
                            print("Maximum JSON parsing retry attempts reached, skipping this sample")
                            return {
                                **sample,
                                "judge_output": None
                            }
                    elif "Error code: 429" in str(e):
                        print("Rate limit reached, retrying after 8 seconds...")
                        time.sleep(8)  
                        continue
                    else:
                        raise

            result_dict = {
                **sample,
                "judge_output": judge_output
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




