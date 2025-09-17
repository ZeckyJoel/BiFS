from torch.utils.data import Dataset
from collections import Counter
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from utils.utils import *


def load_okvqa(data_path):
    entries = []
    with open(data_path, 'r', encoding='utf-8') as fp:
        okvqa_data = json.load(fp)
    
    for sample in okvqa_data:

        base_entry = {
            "img_id": str(sample['image_id']).zfill(12),
            "question_id": sample['question_id'],
            "question": sample['question'],
            "answer_type": sample['answer_type'],
            "question_type": sample['question_type'],
            "confidence": sample['confidence'],
            "direct_answers": sample['direct_answers'],

            "key_information": sample['key_information'],
            "key_reason": sample['key_reason'],
            "knowledge": sample['knowledge']
        }
        
        entries.append(base_entry)
    
    return entries

class OKVQADataset(Dataset):
    def __init__(self, data_path, img_path, split):

        self.split = split
        self.data = load_okvqa(data_path)
        self.image_processor = image_transform(image_size=224)
        
        self.questions = []
        self.direct_answers = []
        self.open_answer_texts = []
        self.image_ids = []
        self.question_ids = []
        
        self.img_path = img_path

        self.key_information = []
        self.key_reason = []
        self.knowledge = []

        for data in self.data:
            question_id = data['question_id']
            image_file = data['img_id']
            question = data['question']
            direct_answers = data['direct_answers']
            most_ans = Counter(direct_answers).most_common()[0][0]

            self.image_ids.append(image_file)
            self.question_ids.append(question_id)
            self.questions.append(question)
            self.open_answer_texts.append(most_ans) 
            self.direct_answers.append(direct_answers)

            self.key_information.append(data['key_information'])
            self.key_reason.append(data['key_reason'])
            self.knowledge.append(data['knowledge'])

    def __len__(self):
        """returns the length of dataframe"""
        return len(self.image_ids)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        question_id = str(self.question_ids[index])
        question = str(self.questions[index])
        direct_answers = str(self.direct_answers[index])
        open_answer_text = str(self.open_answer_texts[index])
        image_file = self.image_ids[index]
        pixel_values = (self.image_processor(load_image(f"{self.img_path}/COCO_{self.split}2014_{image_file}.jpg"))) # [3, 224, 224]

        key_information = self.key_information[index]
        key_reason = self.key_reason[index]
        knowledge = self.knowledge[index]

        return {
            "image_ids": image_file,
            "question_ids": question_id,
            "pixel_values": pixel_values,
            "questions": question,
            "open_answer_texts": open_answer_text,
            "direct_answers_texts": direct_answers,
            
            "key_information": key_information,
            "key_reason": key_reason,
            "knowledge": knowledge
        }
