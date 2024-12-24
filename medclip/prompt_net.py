import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import AutoTokenizer

from medclip import constants, MedCLIPModel, MedCLIPVisionModelViT


def exists(val):
    return val is not None


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x_q, x_kv=None, **kwargs):
        x_q = self.norm(x_q)

        if exists(x_kv):
            x_kv = self.norm_context(x_kv)
        else:
            x_kv = x_q

        return self.fn(x_q, x_kv, x_kv, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(
            self,
            latent_dim,
            kv_dim,
            cross_heads=4,
            seq_dropout_prob=0.
    ):
        super().__init__()
        self.seq_dropout_prob = seq_dropout_prob
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim,
                    nn.MultiheadAttention(latent_dim, num_heads=cross_heads, kdim=kv_dim, vdim=kv_dim,
                                          dropout=seq_dropout_prob, batch_first=True),
                    context_dim=kv_dim),
            FeedForward(latent_dim)])

    def forward(
            self,
            data,
            soft_prompt,
            mask=None,
    ):
        b, *_, device = *data.shape, data.device
        x = soft_prompt
        cross_attn, cross_ff = self.cross_attend_blocks
        x, _ = cross_attn(x, data, key_padding_mask=mask)
        x = cross_ff(x)+x

        return x


class SelfAttention(nn.Module):
    def __init__(
            self,
            depth,
            latent_dim,
            latent_heads=4,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(latent_dim, nn.MultiheadAttention(latent_dim, num_heads=latent_heads, batch_first=True)),
                FeedForward(latent_dim)
            ]))

    def forward(
            self,
            x,
            mask=None
    ):
        # layers

        for self_attn, self_ff in self.layers:
            x = self_attn(x, key_padding_mask=mask)[0] + x
            x = self_ff(x) + x
        return x


class PromptTranslator(nn.Module):
    def __init__(
            self,
            prompt_len,
            prompt_depth,
            prompt_dim = 512,
            depth=4,
            self_heads = 4,
            cross_heads = 4,
            textemb_dim=512,
            device='cuda'
    ):
        super().__init__()
        self.device = device
        self.prompt_len = prompt_len
        self.prompt_depth = prompt_depth
        prompt_dim = prompt_dim
        soft_prompt = torch.empty(prompt_len*prompt_depth, prompt_dim)
        nn.init.normal_(soft_prompt, std=0.02)
        self.soft_prompt = nn.Parameter(soft_prompt)
        self.encoder = CrossAttention(
            latent_dim=prompt_dim,
            kv_dim=textemb_dim,
            cross_heads= cross_heads,
        )
        if depth>0:
            self.transformer = SelfAttention(depth=depth, latent_dim=prompt_dim,latent_heads = self_heads)
        self.depth = depth
    def forward(
            self,
            text_emb,
    ):
        prompt = self.encoder(text_emb, self.soft_prompt)
        if self.depth>0:
            prompt = self.transformer(prompt)
        prompt = prompt.reshape(self.prompt_depth,self.prompt_len,-1)
        return prompt


class PromptLearner:
    def __init__(self,
                 model_dict,
                 lr,
                 train_loader,
                 device,
                 weight_decay,
                 client_id):
        self.model = PromptTranslator(prompt_len=1, prompt_depth=1).to("cuda:0")
        self.model.load_state_dict(model_dict)
        self.train_loader = train_loader
        self.lr = lr
        self.weight_decay = weight_decay
        self.model.to(device)
        dict_path = f'./outputs/models/best_model/person_model_{client_id}.pth'
        if client_id == 'global':
            dict_path = f'./outputs/models/best_model/global_model.pth'
        model_dict = torch.load(dict_path,map_location=torch.device('cpu'))
        self.personal_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT).to("cuda:0")
        self.personal_model.load_state_dict(model_dict)

    def train(self):
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=True)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = torch.nn.MSELoss()

        labels = [ "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
        scaler = GradScaler()
        for i, batch_data in progress_bar:

            optimizer.zero_grad()
            with autocast():
                index = batch_data['labels'][0].item()
                cls = labels[index]
                tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE, local_files_only=True)
                tokenizer.model_max_length = 77

                cls_inputs = tokenizer(cls, truncation=True, max_length=20, padding="max_length", return_tensors='pt')
                text_label = tokenizer(batch_data['reports'], truncation=True, max_length=77, padding="max_length", return_tensors='pt')
                emb1 = self.personal_model.encode_text(cls_inputs['input_ids'], cls_inputs['attention_mask'])
                emb1 = self.model(emb1).reshape(1, 512)
                emb2 = self.personal_model.encode_text(text_label['input_ids'], text_label['attention_mask'])
                loss = criterion(emb1, emb2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            progress_bar.set_postfix({"loss": loss.item()})

    def save(self, client_id):
        save_dir = f'outputs/models/best_model'
        file_path = save_dir + f'/{client_id}_promptNet.pth'
        torch.save(self.model.state_dict(), file_path)
        print(f"Model saved to {file_path}")