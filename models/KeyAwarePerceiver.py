import torch
import torch.nn as nn
from einops import rearrange, repeat
import sys
import os
from transformers import AutoTokenizer, DebertaV2Model

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from utils.utils import *


class KeyAwarePerceiver(nn.Module):
    def __init__(
            self, 
            image_dim=1408, 
            dim=768, 
            num_latents=32,
            depth=2,
            heads=4):
        
        super().__init__()
        
        self.dim = dim

        # DeBERTa v3 文本编码器
        self.tokenizer = AutoTokenizer.from_pretrained('/QA-Prompts/experiments/deberta-v3-base', use_fast=False)
        self.text_encoder = DebertaV2Model.from_pretrained('/QA-Prompts/experiments/deberta-v3-base')

        # 图像特征投影层：使得图像特征与文本编码器输出的维度对齐
        self.image_proj = nn.Linear(image_dim, dim)
        
        # 可学习的潜在查询向量
        self.latents = nn.Parameter(torch.randn(1, num_latents, dim))
        
        # 跨模态注意力块 
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(dim, heads, batch_first=True),
                'cross_attn_text': nn.MultiheadAttention(dim, heads, kdim=dim, vdim=dim, batch_first=True),
                'cross_attn_img': nn.MultiheadAttention(dim, heads, kdim=dim, vdim=dim, batch_first=True),
                'ffn': nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim)
                )
            }) for _ in range(depth)
        ])
        
        self.res_gate = nn.Sequential(
            nn.Linear(dim * 2, dim * 4),
            nn.LayerNorm(dim * 4),  
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Sigmoid()
        )

        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim * 4),
            nn.LayerNorm(dim * 4),   
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Sigmoid()
        )

        self.latent_norm1 = nn.LayerNorm(dim)
        self.latent_norm2 = nn.LayerNorm(dim)
        self.img_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.final_norm = nn.LayerNorm(dim)

        # 注意力池化层
        self.attn_pool = nn.MultiheadAttention(dim, heads, batch_first=True)

        # 将 768 维的特征映射到 LLaMA/Vicuna 模型的输入维度（4096）
        self.llm_proj = nn.Linear(768, 4096)
        # 权重参数从预训练文件加载
        self.llm_proj.load_state_dict(torch.load("/QA-Prompts/experiments/llm_proj_vicuna.pth", map_location='cpu'))

        # 冻结文本编码器参数
        for name, param in self.text_encoder.named_parameters():
            param.requires_grad = False
        self.text_encoder.eval()

    def forward(self, image_embeds, key_information):

        # 1. 提取关键信息文本特征
        text_inputs = self.tokenizer(
            key_information, 
            padding='longest', 
            return_tensors="pt"
        ).to(image_embeds.device)

        with torch.no_grad():
            text_outputs = self.text_encoder(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask
            )
        text_feats = text_outputs.last_hidden_state 
        
        # 2. 处理图像特征
        img_feats = self.image_proj(image_embeds)  # [B, H*W, D]
        
        # 3. 初始化可学习查询
        latents = self.latents.expand(image_embeds.size(0), -1, -1).clone() # [B, 32, D]
        
        # 4. 跨模态交互
        for block in self.blocks:
            # 保存当前图像特征作为残差基础
            img_residual = img_feats.clone()

            # step1: 交叉注意力（latents→关键信息）
            latents_residual = latents.clone() 
            latents, _ = block['cross_attn_text'](
                query=latents,
                key=text_feats,
                value=text_feats
            )
            latents = self.latent_norm1(latents_residual + latents)

            # 自注意力（latents内部关系建模）
            latents_residual = latents.clone() 
            latents, _ = block['self_attn'](
                latents, latents, latents
            )
            latents = self.latent_norm2(latents_residual + latents)
            
            # 交叉注意力（图像→latents）
            img_attended, _ = block['cross_attn_img'](
                query=img_feats,
                key=latents,
                value=latents
            )
            img_attended = self.img_norm(img_feats + img_attended)  
                        
            # 门控特征融合
            gate = self.res_gate(torch.cat([img_residual, img_attended], dim=-1))
            img_feats = gate * img_residual + (1 - gate) * img_attended

            # FFN
            ffn_residual = img_feats.clone()
            img_feats = block['ffn'](img_feats)  
            img_feats = self.ffn_norm(ffn_residual + img_feats) 

        # 注意力池化将img_feats特征重建
        img_pooled, _ = self.attn_pool(
            query=latents,      
            key=img_feats,     
            value=img_feats    
        )  # [B, (256) -> 32, 768]
        img_pooled = self.final_norm(img_pooled)

        # 门控融合更新latents：结合latents和池化后的图像特征
        final_gate = self.gate(torch.cat([latents, img_pooled], dim=-1))
        latents = final_gate * latents + (1 - final_gate) * img_pooled

        query_tokens = self.llm_proj(latents)
        
        return query_tokens  # [B, N, D->4096]


if __name__ == '__main__':

    model = KeyAwarePerceiver(depth=4)
    image_embeds = torch.randn(2, 256, 1408)
    key_information = ['11', 
                       'Key information:\n- Tag: Station Structure\n  Attributes: architecture: arched design with large windows, roof: high arched ceiling with metal framework\n  Caption: The Station Structure provides shelter and an organized environment for rail operations.']
    
    query_tokens = model(image_embeds, key_information)
    print(query_tokens.shape) 