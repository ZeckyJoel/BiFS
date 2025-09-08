from torch.utils.data import Dataset
from collections import Counter
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from utils.utils import *


def load_aokvqa(data_path):
    entries = []
    with open(data_path, 'r', encoding = 'utf-8') as fp:
        aokvqa_data = json.load(fp)

    for sample in aokvqa_data:

        base_entry = {
            "img_id": str(sample['image_id']).zfill(12), 
            "question_id": sample['question_id'],
            "question": sample['question'],
            "choices": sample['choices'],
            "correct_choice_idx": sample['correct_choice_idx'],
            "direct_answers": sample['direct_answers'],
            "rationales": sample['rationales'],

            "key_information": sample['key_information'],
            "key_reason": sample['key_reason'],
            "DA_knowledge": sample['DA_knowledge'],
            "MC_knowledge": sample['MC_knowledge']
        }

        entries.append(base_entry)

    return entries

class AOKVQADataset(Dataset):
    def __init__(self, data_path, img_path):

        self.data = load_aokvqa(data_path)
        self.image_processor = image_transform(image_size=224)

        self.option_0 = []
        self.option_1 = []
        self.option_2 = []
        self.option_3 = []
        self.questions = []
        self.direct_answers = []
        self.open_answer_texts = []
        self.mc_answer_texts = []
        self.image_ids = []
        self.question_ids = []

        self.img_path = img_path

        self.key_information = []
        self.key_reason = []
        self.DA_knowledge = []
        self.MC_knowledge = []
        
        for data in self.data:
            question_id = data['question_id']
            image_file = data['img_id']
            question = data['question']
            direct_answers = data['direct_answers']
            most_ans = Counter(direct_answers).most_common()[0][0]
            choices = data['choices']
            correct_choice_idx = data['correct_choice_idx']

            self.image_ids.append(image_file)
            self.question_ids.append(question_id)
            self.questions.append(question)

            self.direct_answers.append(direct_answers)
            self.open_answer_texts.append(most_ans) 
            self.mc_answer_texts.append(choices[correct_choice_idx])
            
            self.option_0.append(choices[0])
            self.option_1.append(choices[1])
            self.option_2.append(choices[2])
            self.option_3.append(choices[3])

            self.key_information.append(data['key_information'])
            self.key_reason.append(data['key_reason'])
            self.DA_knowledge.append(data['DA_knowledge'])
            self.MC_knowledge.append(data['MC_knowledge'])

    def __len__(self):
        """returns the length of dataframe"""
        return len(self.image_ids)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        question_id = str(self.question_ids[index])
        question = str(self.questions[index])
        option_0 = self.option_0[index]
        option_1 = self.option_1[index]
        option_2 = self.option_2[index]
        option_3 = self.option_3[index]
        direct_answers = str(self.direct_answers[index])
        open_answer_text = str(self.open_answer_texts[index])
        mc_answer_text = str(self.mc_answer_texts[index])
        image_file = self.image_ids[index]
        pixel_values = (self.image_processor(load_image(self.img_path + f"/{image_file}.jpg"))) # [3, 224, 224]
        
        key_information = self.key_information[index]
        key_reason = self.key_reason[index]
        DA_knowledge = self.DA_knowledge[index]
        MC_knowledge = self.MC_knowledge[index]
        
        return {
            "image_ids": image_file,
            "question_ids": question_id,
            "pixel_values": pixel_values,
            "questions": question,
            "option_0": option_0,
            "option_1": option_1,
            "option_2": option_2,
            "option_3": option_3,

            "direct_answers_texts": direct_answers,
            "open_answer_texts": open_answer_text,
            "mc_answer_texts": mc_answer_text,
            
            "key_information": key_information,
            "key_reason": key_reason,
            "DA_knowledge": DA_knowledge,
            "MC_knowledge": MC_knowledge
        }
      
