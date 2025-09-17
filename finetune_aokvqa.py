import numpy as np
import torch
import os
from tqdm import tqdm
import argparse
import time
from utils import *
from torch.cuda.amp import autocast as autocast
import random
from torch.backends import cudnn
from utils.utils import *
from peft import LoraConfig

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_path', type=str, default='/root/lanyun-tmp/output_path')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--coco_path', type=str, default="/QA-Prompts/A-OKVQA-COCO2017")

    parser.add_argument('--bs', type=int, default=6)
    # parser.add_argument('--bs', type=int, default=6) # dubug 
    # parser.add_argument('--eval_bs', type=int, default=24) 
    parser.add_argument('--eval_bs', type=int, default=6) # dubug 

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--model', type=str, default='instruct_blip', choices=['instruct_blip'])
    parser.add_argument('--output_log_path', type=str, default="/QA-Prompts/logs/aok_output_log.txt")
    parser.add_argument('--eval_step', type=int, default=4, help="eval every 1/eval_step epoch")
    parser.add_argument('--dataset', type=str, default='aokvqa', choices=['aokvqa'])
    
    # parser.add_argument('--multi_choices', type=bool, default=True)
    parser.add_argument('--multi_choices', type=bool, default=False)
    
    parser.add_argument('--lr', type=float, default=2e-5)
    # parser.add_argument('--lr', type=float, default=1e-5)
    
    parser.add_argument('--use_KeyReason', type=bool, default=True)
    # parser.add_argument('--use_KeyReason', type=bool, default=False)

    parser.add_argument('--use_KeyInfo', type=bool, default=True)
    # parser.add_argument('--use_KeyInfo', type=bool, default=False)

    # parser.add_argument('--use_knowledge', type=bool, default=False)
    parser.add_argument('--use_knowledge', type=bool, default=True)

    # parser.add_argument('--use_lora', type=bool, default=False)
    parser.add_argument('--use_lora', type=bool, default=True)

    args = parser.parse_args()

    return args

def init_seeds(seed=42, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:  
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  
        cudnn.deterministic = False
        cudnn.benchmark = True

def prepare_inputs_outputs(args, data):

    questions = data["questions"] 
    open_answer_texts = data["open_answer_texts"] 
    key_informations = data["key_information"]
    key_reasons = data["key_reason"]

    if args.multi_choices:
        options_0 = data["option_0"]
        options_1 = data["option_1"]
        options_2 = data["option_2"]
        options_3 = data["option_3"]
        possible_options = [f'Possible options: {option_0}, {option_1}, {option_2}, {option_3}' for option_0, option_1, option_2, option_3 in zip(options_0, options_1, options_2, options_3)]
    else:
        possible_options = ['' for _ in range(len(questions))]

    if args.use_KeyReason:        
        text_input = [
            f"Question: {question}\n\n{key_reason}\n\n{options}Output:" 
            for question, key_reason, options in zip(questions, key_reasons, possible_options)
        ]
    else:
        text_input = [
            f"Question: {question}\n\n{options}Output:"
            for question, options in zip(questions, possible_options)
        ]

    if args.use_knowledge:
        input_prefix = "Answer in format: Because of [knowledge], so the answer is [short answer].\n\n"
        text_input = [input_prefix + s for s in text_input]

        if args.multi_choices:
            # MC_knowledge_list = data["MC_knowledge"]
            MC_knowledge_list = [s.rstrip('.') for s in data["MC_knowledge"]]
            text_output = [
                f"Because of {knowledge}, so the answer is {answer}." 
                for knowledge, answer in zip(MC_knowledge_list, open_answer_texts)
            ]
        else:
            # DA_knowledge_list = data["DA_knowledge"]
            DA_knowledge_list = [s.rstrip('.') for s in data["DA_knowledge"]]
            text_output = [
                f"Because of {knowledge}, so the answer is {answer}." 
                for knowledge, answer in zip(DA_knowledge_list, open_answer_texts)
            ]
    else:
        text_output = data["open_answer_texts"]

    direct_answers_texts = data["direct_answers_texts"]

    return text_input, text_output, questions, direct_answers_texts

@torch.no_grad()
def eval(args, val_loader, model):
    model.eval()
    torch.cuda.empty_cache() 
    val_loss = 0
    val_knowledge_loss = 0
    val_answer_loss = 0
    val_vqa_score = 0

    for step, data in enumerate(tqdm(val_loader)):
        text_input, text_output, questions, direct_answers_texts = prepare_inputs_outputs(args, data)
        samples = {
                "text_input": text_input,
                "text_output": text_output,
                "questions": questions,
                "pixel_values": data["pixel_values"].to(args.device),
                "key_information": data["key_information"]
            }
        generate_kwargs = {
            "do_sample": True,
            "num_beams": 2, 
            "min_length": 1,
            "num_return_sequences": 1,
            "max_new_tokens": 256,
            "temperature":0.7,
            "top_p":0.9,
            }
        with torch.cuda.amp.autocast(enabled=True, dtype=model.dtype): 
            with torch.no_grad():
                outputs = model(samples)
                pred_texts = model.generate(samples, **generate_kwargs)

                answer_pattern = re.compile(r"answer is (.*?)(?=\.)") 
                preds_answers = []
                for pred in pred_texts:
                    match = answer_pattern.search(pred)
                    if match:
                        extracted_answer = match.group(1).strip()
                        preds_answers.append(extracted_answer)
                    else:
                        preds_answers.append(pred.split("answer is")[-1].strip() if "answer is" in pred else pred) 

        loss = outputs['loss']
        loss_knowledge = outputs['loss_knowledge']
        loss_answer = outputs['loss_answer']
        val_loss += loss.item() 
        val_knowledge_loss += loss_knowledge.item()
        val_answer_loss += loss_answer.item()
        val_vqa_score += compute_vqa_score(bs = args.eval_bs, direct_answers_texts = direct_answers_texts, preds = preds_answers)

        if step<=5:
            for i in range(len(text_input)):
                print()
                print("---------------------eval-------------------------")
                print("image_ids: " + data["image_ids"][i] + "  question_ids: " + data["question_ids"][i])
                print("---------------------input-------------------------")
                print(text_input[i])
                print("---------------------preds-------------------------")
                print(pred_texts[i])
                print("--------------------answers------------------------")
                print(text_output[i])
                print()

    val_loss = round(val_loss/len(val_loader), 4)
    val_knowledge_loss = round(val_knowledge_loss / len(val_loader), 4)
    val_answer_loss = round(val_answer_loss / len(val_loader), 4)
    val_vqa_score = round(val_vqa_score/len(val_loader), 4)
    torch.cuda.empty_cache()
    model.train()
    return val_loss, val_knowledge_loss, val_answer_loss, val_vqa_score

def train(args, train_dataset, val_dataset, model):

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.bs, 
        shuffle=True, 
        pin_memory=True,
        drop_last=True,
        num_workers=4  
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.eval_bs,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        num_workers=4
    )

    optimizer = torch.optim.AdamW(filter(lambda p : p.requires_grad, model.parameters()), lr = args.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0)

    max_score = 0
    save_socre = 0.69

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epoch):
        model.train()
        start = time.time()
        train_loss = 0
        train_knowledge_loss = 0
        train_answer_loss = 0
        
        for step, data in enumerate(tqdm(train_loader)):
            text_input, text_output, questions, direct_answers_texts = prepare_inputs_outputs(args, data)
            samples = {
                    "text_input": text_input,
                    "text_output": text_output,
                    "questions": questions,
                    "pixel_values": data["pixel_values"].to(args.device),
                    "key_information": data["key_information"]
                }

            with torch.cuda.amp.autocast(enabled=True, dtype=model.dtype): 
                outputs = model(samples) 
                with torch.no_grad():
                    pred_texts = ['N/A' for i in range(args.bs)]

            loss = outputs['loss']
            train_loss += loss.item()
            train_knowledge_loss += outputs['loss_knowledge'].item()
            train_answer_loss += outputs['loss_answer'].item()
            
            if epoch <= 4 and step % (len(train_loader) // args.eval_step) == 0 and step > 0:     
                torch.cuda.empty_cache()

            if epoch >= 5 and step % (len(train_loader) // args.eval_step) == 0 and step > 0:                
                val_loss, val_knowledge_loss, val_answer_loss, val_vqa_score = eval(args, val_loader, model)

                step_log = 'epoch:{}/{} step:{}  val_loss:{}(K:{},A:{})  val_vqa_score:{}'.format(
                    epoch + 1, args.epoch, step, 
                    val_loss, 
                    val_knowledge_loss,
                    val_answer_loss, 
                    val_vqa_score
                )
                print(step_log)

                if args.output_log_path:
                    with open(args.output_log_path, 'a') as f:
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        f.write(f"[{timestamp}] {step_log}\n")

                if (val_vqa_score >= max_score and val_vqa_score>save_socre):    
                    max_score = val_vqa_score
                    torch.save(
                        model.state_dict(), 
                        f'{args.experiment_path}/model_epoch{epoch+1}_score{val_vqa_score:.4f}.pth'
                    )

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss = round(train_loss/len(train_loader), 4)
        val_loss, val_knowledge_loss, val_answer_loss, val_vqa_score = eval(args, val_loader, model)
        
        end = time.time()

        epoch_log = 'epoch:{}/{}  time:{}h  lr:{}  batchsize:{}  train_loss:{}(K:{},A:{})  val_loss:{}(K:{},A:{})  val_vqa_score:{}'.format(
            epoch + 1, args.epoch, str(round((end - start) / 3600, 2)), 
            args.lr, args.bs, 
            train_loss, 
            round(train_knowledge_loss/len(train_loader),4),
            round(train_answer_loss/len(train_loader),4),
            val_loss,
            val_knowledge_loss, 
            val_answer_loss,
            val_vqa_score
        )
        print(epoch_log)
        
        if args.output_log_path:
            with open(args.output_log_path, 'a') as f:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                f.write(f"[{timestamp}] {epoch_log}\n")

        if (val_vqa_score >= max_score and val_vqa_score>save_socre): 
            max_score = val_vqa_score
            torch.save(
                model.state_dict(), 
                f'{args.experiment_path}/model_epoch{epoch+1}_score{val_vqa_score:.4f}.pth'
            )

if __name__ == '__main__':
    args = parse_args()

    from datasets.aokvqa_dataset import AOKVQADataset
    if args.dataset == 'aokvqa':
        train_dataset = AOKVQADataset(data_path = "/BiFS/MY_DATA/aok_train.json", img_path = os.path.join(args.coco_path, 'train2017'))
        val_dataset = AOKVQADataset(data_path = "/BiFS/MY_DATA/aok_val.json", img_path = os.path.join(args.coco_path, 'val2017'))

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    if args.model == 'instruct_blip':
        from models.blip2_vicuna_instruct_knowledge import Blip2VicunaInstruct
        if args.use_lora:
            lora_config = LoraConfig(
                r=16,                    
                lora_alpha=32,           
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = Blip2VicunaInstruct(
                dtype=torch.float16,
                use_KeyInfo=args.use_KeyInfo,
                lora_config=lora_config 
            ).to(args.device)
        else:
            model = Blip2VicunaInstruct(
                dtype=torch.float16,
                use_KeyInfo=args.use_KeyInfo
            ).to(args.device)

    print("Model Initialized")

    init_seeds(args.seed)

    print(get_parameter_number(model))
    print("dataset: {}  train_num: {}  eval_num: {}  epochs: {}  batch_size_per_gpu: {}  learning_rate: {}".format(args.dataset, len(train_dataset), len(val_dataset), args.epoch, args.bs, args.lr))

    train(args, train_dataset, val_dataset, model)
