import sys
import os
import contextlib
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from transformers import BertTokenizer, LlamaTokenizer
from transformers import BertConfig, Blip2Config
from transformers import LlamaForCausalLM, Blip2VisionModel
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from utils.utils import *
from models.Qformer import BertLMHeadModel
from models.KeyAwarePerceiver import KeyAwarePerceiver

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from peft import LoraConfig, get_peft_model
    
class Blip2VicunaInstruct(nn.Module):
    def __init__(
        self,
        dtype=torch.float16,
        lora_config = None,
        use_KeyInfo = True
    ):
        super().__init__()
        self.dtype = dtype
        self.use_KeyInfo = use_KeyInfo
        
        self.max_input_txt_len = 256 
        self.max_output_txt_len = 256

        self.lora_config = lora_config

        print('loading TextTokenizer')
        self.tokenizer = self.init_tokenizer(truncation_side="left") 

        print('loading ViT')
        blip2_config = Blip2Config.from_pretrained('./experiments/blip2-flan-t5-xl')
        blip2_config.vision_config.torch_dtype = self.dtype
        self.vision_model = Blip2VisionModel(blip2_config.vision_config)
        self.vision_model.load_state_dict(torch.load("./experiments/eva_vit_g.pth", map_location='cpu'))

        print('loading Qformer')
        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token=32, vision_width=blip2_config.vision_config.hidden_size, cross_attention_freq=2)
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None
        self.Qformer.load_state_dict(torch.load("./experiments/qformer_vicuna.pth", map_location='cpu'))
        self.query_tokens = nn.Parameter(torch.load("./experiments/query_tokens_vicuna.pth", map_location='cpu'))

        print('loading Vicuna')
        self.llm_tokenizer = LlamaTokenizer.from_pretrained('./experiments/vicuna-7b', use_fast=False, truncation_side="left")
        self.llm_model = LlamaForCausalLM.from_pretrained('./experiments/vicuna-7b', torch_dtype=self.dtype)
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        if lora_config is not None:
            print("Using LoRA")
            self.lora_config = lora_config
            self.llm_model = get_peft_model(self.llm_model, self.lora_config)
        else:
            print("Frozen vicuna")
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False
            self.llm_model = self.llm_model.eval()

        print('loading llm_proj')
        self.llm_proj = nn.Linear(self.Qformer.config.hidden_size, self.llm_model.config.hidden_size)
        self.llm_proj.load_state_dict(torch.load("./experiments/llm_proj_vicuna.pth", map_location='cpu'))

        if self.use_KeyInfo:
            print("loading KeyAwarePerceiver")
            self.KeyAwarePerceiver = KeyAwarePerceiver()

        print("Frozen ViT")
        for name, param in self.vision_model.named_parameters():
            param.requires_grad = False
        self.vision_model = self.vision_model.eval()

    @property
    def device(self):
        return list(self.parameters())[0].device
    
    def init_tokenizer(self, truncation_side="right"):
        tokenizer = BertTokenizer.from_pretrained("./experiments/bert-base-uncased", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def maybe_autocast(self):
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=self.dtype)
        else:
            return contextlib.nullcontext()

    def init_Qformer(self, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("./experiments/bert-base-uncased")
        encoder_config.encoder_width = vision_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        encoder_config.torch_dtype = self.dtype
        Qformer = BertLMHeadModel(config=encoder_config) 
        query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len

    def encode(self, samples):
        text_input = samples["text_input"]
        pixel_values = samples["pixel_values"]

        with self.maybe_autocast():
            image_embeds = self.vision_model(pixel_values=pixel_values)[0]

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        text_Qformer = self.tokenizer(
            samples['questions'],
            padding='longest',
            truncation=True,
            max_length=self.max_input_txt_len,
            return_tensors="pt",
        ).to(self.device)

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        with self.maybe_autocast():
            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            inputs_llm = self.llm_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])

        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(self.device)

        if self.use_KeyInfo:
            keyInfo_tokens = self.KeyAwarePerceiver(image_embeds, samples['key_information'])
            keyInfo_attens = torch.ones(keyInfo_tokens.size()[:-1], dtype=torch.long).to(self.device)
            inputs_llm = torch.cat([inputs_llm, keyInfo_tokens], dim=1)
            atts_llm = torch.cat([atts_llm, keyInfo_attens], dim=1)

        return inputs_llm, atts_llm
    
    def forward(self, samples, weight_knowledge=4, weight_answer=8):

        # 1. 编码图像和文本输入，获取多模态嵌入和注意力掩码
        inputs_llm, atts_llm = self.encode(samples)
        # print(f"inputs_llm: {inputs_llm.size()}, atts_llm: {atts_llm.size()}")

        # 2. 对输入文本进行tokenize
        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            samples["text_input"],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_input_txt_len,
        ).to(inputs_llm.device)
        # print(f"text_input_tokens.input_ids: {text_input_tokens.input_ids.size()}")
        # print(f"text_input_tokens.attention_mask: {text_input_tokens.attention_mask.size()}")

        # 3. 对输出文本进行tokenize (添加EOS标记)
        self.llm_tokenizer.truncation_side = 'right'
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in samples['text_output']],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(inputs_llm.device)
        # print(f"text_output_tokens.input_ids: {text_output_tokens.input_ids.size()}")
        # print(f"text_output_tokens.attention_mask: {text_output_tokens.attention_mask.size()}")

        # 4. 拼接输入输出token
        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )
        # print(f"concat llm_tokens['input_ids']: {llm_tokens['input_ids'].size()}")
        # print(f"concat llm_tokens['attention_mask']: {llm_tokens['attention_mask'].size()}")
        # print(f"input_part_targets_len: {input_part_targets_len}")

        # 5. 构建训练目标 (将padding位置设置为-100以忽略损失计算)
        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
        )
        # print(f"Targets before input masking: shape={targets.size()} | padding positions: {(targets == -100).sum()}")

        # 6. 将输入部分的目标token设置为-100 (仅计算输出部分的损失)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100
        # print(f"Targets after input masking: shape={targets.size()} | non-pad tokens: {(targets != -100).sum()}")

        # 7. 混合精度计算上下文
        with self.maybe_autocast():
            # 8. 构建空目标掩码 (用于多模态嵌入部分的loss mask)
            empty_targets = (
                torch.ones(atts_llm.size(), dtype=torch.long).to(inputs_llm.device).fill_(-100)
            )
            # print(f"empty_targets: {empty_targets.size()}")

            # 9. 拼接多模态和文本目标
            targets = torch.cat([empty_targets, targets], dim=1)
            # print(f"Targets after multimodal concat: shape={targets.size()} | non-pad tokens: {(targets != -100).sum()}")

            # 10. 获取文本嵌入并拼接多模态嵌入
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
            # print(f"Text inputs_embeds: {inputs_embeds.size()}")

            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)
            # print(f"Final inputs_embeds: {inputs_embeds.size()}")
            # print(f"Final attention_mask: {attention_mask.size()}")

        # 获取知识部分的 mask
        knowledge_targets = targets.clone()

        # 存储有效的知识起止位置
        knowledge_positions = []

        for i, output_text in enumerate(samples['text_output']):
            try:
                # print(f"\n--- Sample {i} ---")
                # print(f"Output text: \"{output_text[:50]}...\"")

                # 分割知识部分和答案部分
                split_token = "so the answer is"
                output_text = output_text.strip()

                idx = output_text.index(split_token) 
                knowledge_part = output_text[:idx].strip() # 知识部分
                # answer_part = output_text[idx:].strip() # 答案部分
                # print(f"Knowledge part: \"{knowledge_part[:50]}...\"")
                # print(f"Answer part: \"{answer_part[:50]}...\"")

                # 重新 tokenize 这两个部分
                knowledge_ids = self.llm_tokenizer(knowledge_part, return_tensors="pt", add_special_tokens=False).input_ids[0].to(inputs_llm.device)
                # answer_ids = self.llm_tokenizer(answer_part, return_tensors="pt", add_special_tokens=False).input_ids[0].to(inputs_llm.device)
                # print(f"Knowledge_ids len: {len(knowledge_ids)}, content: {knowledge_ids[:3]}...")
                # print(f"Answer_ids len: {len(answer_ids)}, content: {answer_ids[:3]}...")

                # 定位目标起始位置
                non_pad_indices = (targets[i] != -100).nonzero(as_tuple=True)[0]
                start_idx = non_pad_indices.tolist() if non_pad_indices.numel() > 0 else [] # 获取第一个非-100的位置
                
                if not start_idx:
                    continue

                label_start = start_idx[0]
                # print(f"First non-pad index: {label_start} (token_id={targets[i][label_start].item()})")

                # 记录知识部分在原序列中的起止位置
                start_pos = label_start
                end_pos = label_start + len(knowledge_ids)
                knowledge_positions.append((start_pos, end_pos))
                # print(f"Knowledge part positions: start={start_pos}, end={end_pos}")

            except Exception as e:
                knowledge_targets[i][:] = -100  # 异常情况下屏蔽知识部分
                knowledge_positions.append((None, None))

        with self.maybe_autocast():
            # print("\n--- Starting model forward ---")
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )
            logits = outputs.logits  # 获取完整预测logits
            # print(f"Model output logits: {logits.size()}")

            valid_len = logits.size(1)
            # print(f"Sequence length: {valid_len}")

            if valid_len <= 1:
                # 特殊处理超短序列（直接返回零损失）
                return {"loss": torch.tensor(0.0), "loss_knowledge": torch.tensor(0.0), "loss_answer": torch.tensor(0.0)}
        
            # 执行shift
            shift_logits = logits[:, :-1, :]  # [batch, valid_len-1, vocab]
            shift_targets = targets[:, 1:]    # [batch, valid_len-1]

            # 创建shift后的知识/答案掩码
            shift_knowledge = torch.full_like(shift_targets, -100)
            shift_answer = torch.full_like(shift_targets, -100)
            # print(f"Shift_logits: {shift_logits.size()}")
            # print(f"Shift_targets: {shift_targets.size()}")

            # 重新对齐掩码位置
            for i, (start, end) in enumerate(knowledge_positions):
                if start is None or end is None:
                    continue  
                
                # 计算shift后的新位置
                # 注意：原位置从start到end-1 -> shift后对应位置start-1到end-2
                new_start = max(0, start - 1)  # 避免负索引
                new_end = min(end - 1, valid_len - 1)  # 避免越界

                # print(f"Sample {i}: Original ({start}, {end}) -> Shifted ({new_start}, {new_end})")
                
                if new_start < new_end:  # 确保有效区间
                    # 知识部分（原knowledge_targets[start:end]）
                    shift_knowledge[i, new_start:new_end] = shift_targets[i, new_start:new_end]
                    # 答案部分（原answer_targets[end:]）
                    shift_answer[i, new_end:] = shift_targets[i, new_end:]

                    # print(f"  Knowledge tokens: {torch.sum(shift_knowledge[i] != -100).item()}")
                    # print(f"  Answer tokens: {torch.sum(shift_answer[i] != -100).item()}")  

            # print(f"Final shift_knowledge: non-pad tokens={torch.sum(shift_knowledge != -100).item()}")
            # print(f"Final shift_answer: non-pad tokens={torch.sum(shift_answer != -100).item()}")                  

            # 计算知识部分损失
            loss_knowledge = self.compute_masked_loss(
                shift_logits, 
                shift_knowledge,
            ) 
            
            # 计算答案部分损失
            loss_answer = self.compute_masked_loss(
                shift_logits, 
                shift_answer,
            ) 
            
            loss = weight_knowledge * loss_knowledge + weight_answer * loss_answer

        return {
            "loss": loss,
            "loss_knowledge": loss_knowledge,
            "loss_answer": loss_answer
        }

    def compute_masked_loss(self, logits, targets):
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            logits.reshape(-1, logits.size(-1)), 
            targets.reshape(-1)                   
        )
        return loss

    @torch.no_grad()
    def generate(
        self,
        samples,
        **generate_kwargs
    ):
        inputs_llm, atts_llm = self.encode(samples)

        self.llm_tokenizer.padding_side = "left"

        llm_tokens = self.llm_tokenizer(
            samples["text_input"],
            padding="longest",
            return_tensors="pt"
        ).to(inputs_llm.device)

        with self.maybe_autocast():
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **generate_kwargs
            )

        outputs[outputs == 0] = 2
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip().replace('<s> ', '') for text in output_text]

        return output_text