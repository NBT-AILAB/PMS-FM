import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
from torchvision import models, transforms
import torch.nn.functional as F
from medclip import constants
from transformers import BertModel, BertTokenizer
ids = {"client_1": 0, "client_2": 1, "client_3": 2, "client_4": 3}
beta = 0.4
class TextModel(nn.Module):
    def __init__(self):
        model_name = constants.BERT_TYPE
        super(TextModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.tokenizer = AutoModel.from_pretrained(model_name)
        for param in self.model.parameters():
            param.requires_grad = False
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Use the last hidden state which has the shape [batch_size, sequence_length, hidden_size]
        return outputs['pooler_output']


class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.vit_type = constants.VIT_TYPE
        self.model = AutoModel.from_pretrained(self.vit_type)
        for param in self.model.parameters():
            param.requires_grad = False
    def forward(self, pixel):
        output = self.model(pixel)
        return output['pooler_output']


class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(CrossAttention, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value):
        # Cross-attention
        attn_output, _ = self.attn(query, key, value)
        # Dropout, normalization and residual connection
        attn_output = self.dropout(attn_output)
        output = self.norm(query + attn_output)
        return output


class MLPFusion_Mdoel(nn.Module):
    def __init__(self,
                 text_model = None,
                 image_model = None,
                 num_classes = 1,
                 ):
        text_model = TextModel()
        image_model = ImageModel()
        super(MLPFusion_Mdoel, self).__init__()
        self.text_model = text_model if text_model is not None else TextModel()
        self.image_model = image_model if image_model is not None else ImageModel()
        self.fc1 = nn.Linear(768 + 768, 512)
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512,512)
        self.fc4 = nn.Linear(512, num_classes)

    def forward(self, pixel, input_ids, attention_mask):
        text_features = self.text_model(input_ids = input_ids, attention_mask = attention_mask)
        image_features = self.image_model(pixel)
        combined_features = torch.cat((text_features, image_features), dim=1)
        x = torch.relu(self.fc1(combined_features))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        output = F.softmax(x, dim=1)
        return output

class CAFusion_Mdoel(nn.Module):
    def __init__(self,
                 text_model=None,
                 image_model=None,
                 num_classes=1,
                 d_model=768,
                 num_heads=8,
                 num_layers=6
                 ):
        text_model = TextModel()
        image_model = ImageModel()
        super(CAFusion_Mdoel, self).__init__()
        self.text_model = text_model if text_model is not None else TextModel()
        self.image_model = image_model if image_model is not None else ImageModel()
        self.cross_attention = CrossAttention(d_model=d_model, num_heads=num_heads)
        self.cross_attention_layers = nn.ModuleList(
            [CrossAttention(d_model=d_model, num_heads=num_heads) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, pixel, input_ids, attention_mask):
        text_features = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        image_features = self.image_model(pixel)
        for cross_attention in self.cross_attention_layers:
            text_features = cross_attention(text_features, image_features, image_features)
        cross_attn_output = text_features.squeeze(1)
        x = self.fc(cross_attn_output)
        output = F.softmax(x, dim=1)
        return output

    def inference(self, text_features, image_features, client_id):
        for cross_attention in self.cross_attention_layers:
            text_features = cross_attention(text_features, image_features, image_features)
        cross_attn_output = text_features.squeeze(1)
        x = self.fc(cross_attn_output)
        outputs = F.softmax(x, dim=1).cpu().detach().numpy()
        outputs[0][ids[client_id]] += beta
        return outputs
class PromptLearner(nn.Module):
    def __init__(self, n_classes):
        super(PromptLearner, self).__init__()
        model_name = constants.BERT_TYPE
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.tokenizer = AutoModel.from_pretrained(model_name)
        for param in self.model.parameters():
            param.requires_grad = False
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[1]
        output = self.drop(pooled_output)
        return self.out(output)