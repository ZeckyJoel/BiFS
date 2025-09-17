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

        self.tokenizer = AutoTokenizer.from_pretrained('/QA-Prompts/experiments/deberta-v3-base', use_fast=False)
        self.text_encoder = DebertaV2Model.from_pretrained('/QA-Prompts/experiments/deberta-v3-base')

        self.image_proj = nn.Linear(image_dim, dim)
        
        self.latents = nn.Parameter(torch.randn(1, num_latents, dim))
        
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

        self.attn_pool = nn.MultiheadAttention(dim, heads, batch_first=True)

        self.llm_proj = nn.Linear(768, 4096)
        self.llm_proj.load_state_dict(torch.load("/QA-Prompts/experiments/llm_proj_vicuna.pth", map_location='cpu'))

        for name, param in self.text_encoder.named_parameters():
            param.requires_grad = False
        self.text_encoder.eval()

    def forward(self, image_embeds, key_information):

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
        
        img_feats = self.image_proj(image_embeds)  # [B, H*W, D]
        
        latents = self.latents.expand(image_embeds.size(0), -1, -1).clone() # [B, 32, D]
        
        for block in self.blocks:
            img_residual = img_feats.clone()

            latents_residual = latents.clone() 
            latents, _ = block['cross_attn_text'](
                query=latents,
                key=text_feats,
                value=text_feats
            )
            latents = self.latent_norm1(latents_residual + latents)

            latents_residual = latents.clone() 
            latents, _ = block['self_attn'](
                latents, latents, latents
            )
            latents = self.latent_norm2(latents_residual + latents)
            
            img_attended, _ = block['cross_attn_img'](
                query=img_feats,
                key=latents,
                value=latents
            )
            img_attended = self.img_norm(img_feats + img_attended)  
                        
            gate = self.res_gate(torch.cat([img_residual, img_attended], dim=-1))
            img_feats = gate * img_residual + (1 - gate) * img_attended

            # FFN
            ffn_residual = img_feats.clone()
            img_feats = block['ffn'](img_feats)  
            img_feats = self.ffn_norm(ffn_residual + img_feats) 

        img_pooled, _ = self.attn_pool(
            query=latents,      
            key=img_feats,     
            value=img_feats    
        )  # [B, (256) -> 32, 768]
        img_pooled = self.final_norm(img_pooled)

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