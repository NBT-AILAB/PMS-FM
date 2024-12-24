import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch.nn.functional as F
from medclip import constants, MedCLIPModel, MedCLIPVisionModelViT, PromptClassifier
from medclip.multi_fusion import MLPFusion_Mdoel, CAFusion_Mdoel
from medclip.multi_fusion import PromptLearner
from medclip.prompt_net import PromptTranslator
from medclip.prompts import generate_chexpert_class_prompts
from medclip.dataset import ImageTextContrastiveDataset, ImageTextContrastiveCollator, ZeroShotImageDataset, \
    ZeroShotImageCollator

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONASHSEED'] = str(seed)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def get_batch_size(client_id, method):
    batch_size = constants.batch_size[method][client_id]
    return batch_size


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))  # subtract the max for numerical stability
    return e_x / e_x.sum(axis=0)  # the sum is computed along the only axis (axis=0)


def get_valid_dataloader(client):
    dataset_path = constants.DATASET_PATH
    datalist_path = constants.DATALIST_PATH
    val_data = ZeroShotImageDataset(class_names=constants.CHEXPERT_COMPETITION_TASKS,
                                    dataset_path=dataset_path,
                                    datalist_path=datalist_path,
                                    client=client, )
    val_collate_fn = ZeroShotImageCollator(mode='multiclass')
    val_dataloader = DataLoader(val_data,
                                batch_size=1,
                                collate_fn=val_collate_fn,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=0,
                                )
    return val_dataloader


global_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
select_model = CAFusion_Mdoel(num_classes=constants.SELECT_NUM)
select_dict = torch.load('./outputs/models/select_ca1.pth', map_location=torch.device('cuda:0'))
select_model.load_state_dict(select_dict, False)
select_model.to("cuda:0")
global_dict = torch.load('./outputs/models/fed_avg1.pth', map_location=torch.device('cuda:0'))
global_model.load_state_dict(global_dict, False)
global_model.to("cuda:0")
thd = constants.THRESHOLD
client_ids = ["client_1", "client_2", "client_3", "client_4"]
person_models = {}
prompt_models = {}

for client_id in client_ids:
    person_dict = torch.load(f'./outputs/models/person_model_{client_id}.pth',
                             map_location=torch.device('cuda:0'))
    person_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    person_model.load_state_dict(person_dict, False)
    person_models[client_id] = person_model
prompt_dict = torch.load(f'./outputs/models/global_promptNet.pth',
                         map_location=torch.device('cuda:0'))
prompt_global_model = PromptTranslator(prompt_len=1, prompt_depth=1).to("cuda:0")
prompt_global_model.load_state_dict(prompt_dict)
for client_id in client_ids:
    prompt_dict = torch.load(f'./outputs/models/{client_id}_promptNet.pth',
                             map_location=torch.device('cuda:0'))
    prompt_model = PromptTranslator(prompt_len=1, prompt_depth=1).to("cuda:0")
    prompt_model.load_state_dict(prompt_dict)
    prompt_models[client_id] = prompt_model
for client_id in client_ids:
    person_models[client_id].to("cuda:0")
    prompt_models[client_id].to("cuda:0")


def eval_with_pms(client_id):
    val_data = get_valid_dataloader(client_id)
    pred_label = []
    label_list = []
    tasks = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Pleural Effusion",
    ]
    cnt = 0
    for i, batch_data in enumerate(val_data):
        print(i)
        pixel = batch_data["pixel_values"].to("cuda:0")
        logits = []
        for task in tasks:
            tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE, local_files_only=True)
            tokenizer.model_max_length = 77
            cls_inputs = tokenizer(task, truncation=True, padding=True, return_tensors='pt')
            emb1 = global_model.encode_text(cls_inputs['input_ids'], cls_inputs['attention_mask']).to("cuda:0")
            emb1 = prompt_models[client_id](emb1).reshape(1, 512)
            emb1 = F.pad(emb1, (0, 768 - 512))
            input_ids = batch_data["prompt_inputs"][task]["input_ids"].view(1, -1).to("cuda:0")
            attention_mask = batch_data["prompt_inputs"][task]["attention_mask"].view(1, -1).to("cuda:0")
            text_features = select_model.text_model(input_ids=input_ids, attention_mask=attention_mask)
            image_features = select_model.image_model(pixel)
            text_features = 0.2 * emb1 + 0.8 * text_features
            outputs = select_model.inference(text_features, image_features)
            max_index = np.argmax(outputs)
            person_model = person_models[client_ids[max_index]]
            if np.max(outputs) <= 0.4:
                person_model = global_model
            img_embeds = person_model.encode_image(pixel)
            text_embeds = person_model.encode_text(input_ids, attention_mask)
            logit = []
            img = img_embeds
            txt = text_embeds
            logit = global_model.compute_logits(txt, img).cpu().detach().item()
            logits.append(logit)
        pred = np.argmax(logits)
        pred_label.append(pred)
        label_list.append(batch_data['labels'])
    labels = label_list
    acc = sum(x == y for x, y in zip(pred_label, labels)) / len(val_data.dataset)
    print(f'psm model in {client_id} its acc is {acc.item():.3f}')
    return acc


for client_id in client_ids:
    eval_with_pms(client_id)
