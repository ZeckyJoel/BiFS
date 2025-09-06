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

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from peft import LoraConfig, get_peft_model
    
class Blip2VicunaInstruct(nn.Module):
    def __init__(
        self,
        dtype=torch.float16,
        lora_config = None
    ):
        super().__init__()
        self.dtype = dtype
        
        # self.max_input_txt_len = 512 
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
        # legacy=False:
        # self.llm_tokenizer = LlamaTokenizer.from_pretrained('./experiments/vicuna-7b', use_fast=False, truncation_side="left", legacy=False)
        self.llm_tokenizer = LlamaTokenizer.from_pretrained('./experiments/vicuna-7b', use_fast=False, truncation_side="left")
        self.llm_model = LlamaForCausalLM.from_pretrained('./experiments/vicuna-7b', torch_dtype=self.dtype)
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        # >>>>>>>>>> 添加 LoRA 支持 <<<<<<<<<<
        if lora_config is not None:
            print("Using LoRA")
            self.lora_config = lora_config
            self.llm_model = get_peft_model(self.llm_model, self.lora_config)
        else:
            # 默认冻结原始 LLM 权重
            print("Frozen vicuna")
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False
            self.llm_model = self.llm_model.eval()

        print('loading llm_proj')
        self.llm_proj = nn.Linear(self.Qformer.config.hidden_size, self.llm_model.config.hidden_size)
        self.llm_proj.load_state_dict(torch.load("./experiments/llm_proj_vicuna.pth", map_location='cpu'))

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
        # 根据当前设备（CPU/GPU）决定是否启用混合精度计算（Mixed Precision Training/Inference）。
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=self.dtype)
        else:
            return contextlib.nullcontext()

    def init_Qformer(self, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("./experiments/bert-base-uncased")
        encoder_config.encoder_width = vision_width # 设置 Q-Former 能够处理的图像特征维度（通常是 ViT 输出的 hidden size）
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq # 控制在freq层之间插入交叉注意力模块
        encoder_config.query_length = num_query_token
        encoder_config.torch_dtype = self.dtype
        # 使用修改后的 BertConfig 初始化一个带有语言建模头（LM Head）的 BERT 模型，作为 Q-Former。
        # 这个模型能够接收图像特征并对其进行跨模态注意力操作。
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
        text_input = samples["text_input"] # 从样本中获取输入文本
        pixel_values = samples["pixel_values"] # 从样本中获取图像的像素值

        with self.maybe_autocast():
            image_embeds = self.vision_model(pixel_values=pixel_values)[0] # 利用 ViT 模型将图像编码为特征向量

        # 生成图像特征的 attention mask，全为 1，表示所有位置都有效
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)

        # 初始化 query tokens，扩展预定义的 query tokens 以匹配当前 batch 的大小
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        # 使用 BERT tokenizer 对问题进行编码
        text_Qformer = self.tokenizer(
            samples['questions'],
            padding='longest',
            truncation=True,
            max_length=self.max_input_txt_len,
            return_tensors="pt",
        ).to(self.device)

        # 为 query tokens 创建全1的 attention mask
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)
        # 将 query tokens 的 attention mask 与问题的 attention mask 拼接在一起
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)

        # 使用 Q-Former 模型对图像特征和查询向量进行跨模态注意力操作
        with self.maybe_autocast():
            query_output = self.Qformer.bert(
                text_Qformer.input_ids, # 输入QFormer的文本特征
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds, # 图像特征
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            # 通过线性投影层 (llm_proj) 将输出映射到 LLM 的输入维度
            inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])

        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(self.device)

        return inputs_llm, atts_llm
    
    def forward(self, samples):
        # 调用 encode() 获取图像-文本融合表示
        inputs_llm, atts_llm = self.encode(samples)

        # 对输入文本进行分词（Tokenization）
        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            samples["text_input"],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_input_txt_len,
        ).to(inputs_llm.device)

        # print("input_ids shape:", text_input_tokens['input_ids'].shape)  # [batch_size, seq_len]
        # print("attention_mask shape:", text_input_tokens['attention_mask'].shape)  # [batch_size, seq_len]

        # 对输出文本进行分词（带 EOS 标记）
        self.llm_tokenizer.truncation_side = 'right'
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in samples['text_output']],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(inputs_llm.device)

        #  将输入和输出的 token ID 及 attention mask 拼接在一起，构建完整的上下文序列，便于语言模型训练
        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        # do not apply loss to the padding
        # 构建训练用的目标标签（labels），并将所有 pad token 的位置设置为 -100
        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
        )

        # do not apply loss to the text input (i.e., instruction)
        # 屏蔽输入部分的标签，不对输入文本（如问题或指令）部分计算损失，只对输出部分（如答案）计算损失
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        with self.maybe_autocast():
            # 创建一个与图像嵌入部分等长的全 -100 目标张量，并将其与之前的文本目标拼接，形成完整的标签序列
            empty_targets = (
                torch.ones(atts_llm.size(), dtype=torch.long).to(inputs_llm.device).fill_(-100)
            )
            targets = torch.cat([empty_targets, targets], dim=1)

            # 将 token ID 转换为嵌入向量
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
            # 将图像嵌入（inputs_llm）与文本嵌入拼接
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            # 将图像和文本的 attention mask 拼接
            attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)

        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        loss = outputs.loss

        return {
            "loss": loss
        }

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

        # 将输出中值为 0 的 token 替换为 2，即 EOS（end-of-sequence）标记，解决某些情况下出现无效 token 的问题
        outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # 去除首尾空格，并替换掉多余的 <s> 开头内容，使输出更干净
        output_text = [text.strip().replace('<s> ','') for text in output_text]

        return output_text


