import torch
import torch.nn as nn
from transformers import BertModel

class SBERT(nn.Module):
    def __init__(self, bert_path="distilbert-base-uncased", hidden_dim=768):
        super(SBERT, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.fc = nn.Linear(hidden_dim * 3, 3)  # 3 classes: entailment, contradiction, neutral

    def mean_pooling(self, token_embeds, attention_mask):
        """Mean Pooling for Sentence Representation"""
        in_mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
        return torch.sum(token_embeds * in_mask, 1) / torch.clamp(in_mask.sum(1), min=1e-9)

    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        """Forward Pass"""
        u = self.bert(input_ids_a, attention_mask=attention_mask_a).last_hidden_state
        v = self.bert(input_ids_b, attention_mask=attention_mask_b).last_hidden_state
        
        u_mean = self.mean_pooling(u, attention_mask_a)
        v_mean = self.mean_pooling(v, attention_mask_b)
        
        x = torch.cat([u_mean, v_mean, torch.abs(u_mean - v_mean)], dim=-1)
        logits = self.fc(x)
        return logits  # No softmax needed with CrossEntropyLoss
