import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
This module defines the utility functions for the classification and unlearning framework.

For NPO algorithm, you can check the original version from https://github.com/licong-lin/negative-preference-optimization/blob/main/TOFU/dataloader.py  [line 254] 

Developer name: Xingyi Zhao
Email: xingyi.zhao@usu.edu
Affiliation: Utah State University [Logan, UT, USA]
Last Modified Time: 2026-02-22
'''

def set_seed(random_seed=11):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

################################# Dataset #################################  

class TargetDataset(Dataset):
    def __init__(self, tokenizer, max_len, data):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text = data["text"]
        self.targets = data["label"]
        self.flag = data["poisoned"]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        target = self.targets[item]
        flag = self.flag[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(target, dtype=torch.long),
            "flag": flag
        }

################################# Model ################################# 

class BertClassification(nn.Module):
    def __init__(self, bert, label_num=2):
        super(BertClassification, self).__init__()
        self.bert = bert
        self.classifier = nn.Linear(768, label_num)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask,
                           output_hidden_states=True,
                           output_attentions=True)

        sequence_output = output[0]
        cls_rep = sequence_output[:, 0, :]

        return self.classifier(cls_rep)

class DistilBertClassification(nn.Module):
    def __init__(self, bert, label_num=2):
        super(DistilBertClassification, self).__init__()
        self.bert = bert
        self.classifier = nn.Linear(768, label_num)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask,
                           output_hidden_states=True,
                           output_attentions=True)

        sequence_output = output.last_hidden_state
        cls_rep = sequence_output[:, 0, :]

        return self.classifier(cls_rep)

class LlamaClassification(nn.Module):
    """
    Llama model for classification tasks.
    Pooling options:
    1) first token: hidden state of the first token <s>
    2) last token: hidden state of the last token

    Note: Llama2-7B's tokenizer does not have a [CLS] token, so we use the last token representation for classification (detection).
    """
    def __init__(self, llama_model, label_num=2, pooling="last"):
        super().__init__()
        self.llama = llama_model
        self.hidden_size = self.llama.config.hidden_size
        self.config = self.llama.config
        self.pooling = pooling
        self.classifier = nn.Linear(self.hidden_size, label_num)
        self.classifier = self.classifier.to(llama_model.device)
        self.classifier = self.classifier.to(llama_model.dtype)

    def _pool(self, hidden_states, attention_mask):
        if self.pooling == "first":
            return hidden_states[:, 0, :]  # <s>

        elif self.pooling == "last":
            seq_lengths = attention_mask.sum(1) - 1  # (B,)
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_idx, seq_lengths]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

    def forward(self, 
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                inputs_embeds=None,
                past_key_values=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=False,
                **kwargs):
        out = self.llama(input_ids=input_ids,
                         attention_mask=attention_mask,
                         position_ids=position_ids,
                         inputs_embeds=inputs_embeds,
                         past_key_values=past_key_values,
                         use_cache=use_cache,
                         output_attentions=output_attentions,
                         output_hidden_states=output_hidden_states)
        last_hidden_state = out.last_hidden_state
        sent_rep = self._pool(last_hidden_state, attention_mask)
        return self.classifier(sent_rep)  # (batch, label_num)

def extract_cls_hidden_bert(bert, input_ids, attn_mask, layer_idx=-1):
    """Return CLS vector from a specific layer (no_grad context assumed)."""
    hs = bert(
        input_ids=input_ids,
        attention_mask=attn_mask,
        output_hidden_states=True,
    ).hidden_states[layer_idx][:, 0, :]  # (batch, 768)
    return hs

def rep_extract_bert(bert, loader, device):
    bert.eval()
    rep_vec = []

    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)

            cls_vec = extract_cls_hidden_bert(bert, input_ids, attn_mask, layer_idx=-1)
            rep_vec.append(cls_vec.cpu())

    return torch.cat(rep_vec).numpy()

def extract_cls_hidden_llama(llama, input_ids, attn_mask, layer_idx=-1, pooling="last"):
    """Return last token representation"""
    hs = llama(
        input_ids=input_ids,
        attention_mask=attn_mask,
        output_hidden_states=True,
    ).hidden_states[layer_idx]

    if pooling == "first":
        return hs[:, 0, :]

    elif pooling == "last":
        seq_lengths = attn_mask.sum(1) - 1  # (B,)
        batch_idx = torch.arange(hs.size(0), device=hs.device)
        return hs[batch_idx, seq_lengths]

def rep_extract_llama(llama, loader, device):
    llama.eval()
    rep_vec = []

    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)

            cls_vec = extract_cls_hidden_llama(llama, input_ids, attn_mask, layer_idx=-1)
            rep_vec.append(cls_vec.cpu().float())

    return torch.cat(rep_vec).numpy()

################################# Method ################################# 

def train(model, dataloader, args, device):
    """Standard training loop for the classification model."""
    total_loss = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    for batch in tqdm(dataloader):
        model.train()
        model.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)  # clip the gradients to 1.0

        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(dataloader)  # average batch loss for each epoch

def ga_train(model, retain_loader, forget_loader, args, device):
    """Gradient Ascent Unlearning (GA): ReTain - Forget"""
    clean_loss = 0
    poison_loss = 0

    batches_retain = list(retain_loader)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    for batch_forget in forget_loader:
        model.train()
        model.zero_grad()
        optimizer.zero_grad()

        # Sample retain batch
        batch_retain = random.choice(batches_retain)
        input_ids_retain = batch_retain["input_ids"].to(device)
        attention_mask_retain = batch_retain["attention_mask"].to(device)
        labels_retain = batch_retain["label"].to(device)

        # Forget batch
        input_ids_forget = batch_forget["input_ids"].to(device)
        attention_mask_forget = batch_forget["attention_mask"].to(device)
        labels_forget = batch_forget["label"].to(device)

        logits_retain = model(input_ids=input_ids_retain, attention_mask=attention_mask_retain)
        logits_forget = model(input_ids=input_ids_forget, attention_mask=attention_mask_forget)

        # Calculate loss
        loss_retain = loss_fn(logits_retain, labels_retain)  # clean samples: keep the functioning of the model
        loss_forget = loss_fn(logits_forget, labels_forget)  # poisoned samples: forget the backdoor of the model
        loss = loss_retain - loss_forget

        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)  # clip the gradients to 1.0
        optimizer.step()

        clean_loss += loss_retain.item()
        poison_loss += loss_forget.item()

    return clean_loss / len(forget_loader), poison_loss / len(forget_loader)

def npo_train(model, ref_model, retain_loader, forget_loader, args, device):
    """"Negative Preference Optimization Unlearning (NPO): ReTain - Forget (NPO Term) with a reference model to guide the forgetting process."""
    clean_loss = 0
    poison_loss = 0
    beta = 1  # Same beta as in the origial NPO paper (https://openreview.net/pdf?id=MXLBXjQkmb)

    batches_retain = list(retain_loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    loss_fn_base = torch.nn.CrossEntropyLoss(reduction="none")
    loss_fn_mean = torch.nn.CrossEntropyLoss(reduction="mean")

    for batch_forget in tqdm(forget_loader):
        model.train()
        model.zero_grad()
        optimizer.zero_grad()

        # Sample retain batch
        batch_retain = random.choice(batches_retain)
        input_ids_retain = batch_retain["input_ids"].to(device)
        attention_mask_retain = batch_retain["attention_mask"].to(device)
        labels_retain = batch_retain["label"].to(device)

        # Forget batch
        input_ids_forget = batch_forget["input_ids"].to(device)
        attention_mask_forget = batch_forget["attention_mask"].to(device)
        labels_forget = batch_forget["label"].to(device)

        # Retain loss
        logits_retain = model(input_ids=input_ids_retain, attention_mask=attention_mask_retain)
        loss_retain = loss_fn_mean(logits_retain, labels_retain)

        # NPO loss
        # [Reference model]
        with torch.no_grad():
            ref_model.eval()
            logits_forget_ref = ref_model(input_ids=input_ids_forget, attention_mask=attention_mask_forget)
            loss_forget_ref = loss_fn_base(logits_forget_ref, labels_forget)

        # [Policy model]
        logits_forget = model(input_ids=input_ids_forget, attention_mask=attention_mask_forget)
        loss_forget = loss_fn_base(logits_forget, labels_forget)

        negative_log_ratio = loss_forget - loss_forget_ref
        npo_loss = -F.logsigmoid(beta * negative_log_ratio).mean() * 2 / beta

        loss = loss_retain + npo_loss  # RT + NPO
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)  # clip the gradients to 1.0
        optimizer.step()

        clean_loss += loss_retain.item()
        poison_loss += loss_forget.mean().item()

    return clean_loss / len(forget_loader), poison_loss / len(forget_loader)


def rga_train(model, model_base, logits_ref, retain_loader, forget_loader, args, device):
    """"Robustness Gradient Ascent Unlearning (RGA): ReTain - λ*Forget + Regularization."""
    clean_loss = 0
    poison_loss = 0

    batches_retain = list(retain_loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    for batch_forget, logits_ref_batch in tqdm(zip(forget_loader, logits_ref)):
        model.train()
        model.zero_grad()
        optimizer.zero_grad()

        # Sample retain batch
        batch_retain = random.choice(batches_retain)
        input_ids_retain = batch_retain["input_ids"].to(device)
        attention_mask_retain = batch_retain["attention_mask"].to(device)
        labels_retain = batch_retain["label"].to(device)

        # Forget batch
        input_ids_forget = batch_forget["input_ids"].to(device)
        attention_mask_forget = batch_forget["attention_mask"].to(device)
        labels_forget = batch_forget["label"].to(device)

        logits_retain = model(input_ids=input_ids_retain, attention_mask=attention_mask_retain)
        logits_forget = model(input_ids=input_ids_forget, attention_mask=attention_mask_forget)

        # Adaptive Re-weight Gradient Ascent
        with torch.no_grad():
            weight = adaptive_weight(logits_forget, logits_ref_batch.to(device), scale_power=2.0).item()

        # Calculate loss
        loss_retain = loss_fn(logits_retain, labels_retain)  # clean samples: keep the functioning of the model
        loss_forget = loss_fn(logits_forget, labels_forget)  # poisoned samples: forget the backdoor of the model
        loss = loss_retain - weight * loss_forget

        # Regularization Term
        for param_1, param_2 in zip(model.bert.parameters(), model_base.parameters()):
            loss += 5e-2 * torch.norm(param_1 - param_2).to(device)  # L2 regularization

        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        clean_loss += loss_retain.item()
        poison_loss += loss_forget.item()

    return clean_loss / len(forget_loader), poison_loss / len(forget_loader)

def rga_train_llama(model, model_base, logits_ref, retain_loader, forget_loader, args, device):
    """"Robustness Gradient Ascent Unlearning (RGA): ReTain - λ*Forget + Regularization."""
    clean_loss = 0
    poison_loss = 0

    batches_retain = list(retain_loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    for batch_forget, logits_ref_batch in tqdm(zip(forget_loader, logits_ref)):
        model.train()
        model.zero_grad()
        optimizer.zero_grad()

        # Sample retain batch
        batch_retain = random.choice(batches_retain)
        input_ids_retain = batch_retain["input_ids"].to(device)
        attention_mask_retain = batch_retain["attention_mask"].to(device)
        labels_retain = batch_retain["label"].to(device)

        # Forget batch
        input_ids_forget = batch_forget["input_ids"].to(device)
        attention_mask_forget = batch_forget["attention_mask"].to(device)
        labels_forget = batch_forget["label"].to(device)

        logits_retain = model(input_ids=input_ids_retain, attention_mask=attention_mask_retain)
        logits_forget = model(input_ids=input_ids_forget, attention_mask=attention_mask_forget)

        # Adaptive Re-weight Gradient Ascent
        with torch.no_grad():
            weight = adaptive_weight(logits_forget, logits_ref_batch.to(device), scale_power=2.0).item()

        # Calculate loss
        loss_retain = loss_fn(logits_retain, labels_retain)  # clean samples: keep the functioning of the model
        loss_forget = loss_fn(logits_forget, labels_forget)  # poisoned samples: forget the backdoor of the model
        loss = loss_retain - weight * loss_forget

        # Regularization Term
        for param_1, param_2 in zip(model.llama.parameters(), model_base.llama.parameters()):
            loss += 5e-2 * torch.norm(param_1 - param_2).to(device)  # L2 regularization

        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        clean_loss += loss_retain.item()
        poison_loss += loss_forget.item()

    return clean_loss / len(forget_loader), poison_loss / len(forget_loader)


def adaptive_weight(input_logits, ref_logits, scale_power=1.0):
    input_prob = torch.log_softmax(input_logits, dim=1)
    ref_prob = torch.softmax(ref_logits, dim=1)
    kl_divergence = F.kl_div(input_prob, ref_prob, reduction="batchmean")
    weight = torch.pow(torch.exp(-kl_divergence), scale_power)

    return weight

def compute_logits_ref(ref_model, forget_loader, device):
    logits_ref = []

    with torch.no_grad():
        for batch in tqdm(forget_loader):
            ref_model.eval()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = ref_model(input_ids=input_ids, attention_mask=attention_mask)
            logits_ref.append(logits.detach().cpu())

    return logits_ref

################################# Evaluation ################################# 

def evaluate(model, dataloader, data_name, device):
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            model.eval()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(labels.numpy())

    accuracy = accuracy_score(all_targets, all_predictions)
    cm = confusion_matrix(all_targets, all_predictions)

    print(f"Accuracy on {data_name} data: {accuracy}")
    print(f"Confusion matrix on {data_name} data:\n {cm}")

    if data_name == "SST-2" or data_name == "HSOL":
        LFR_0 = cm[0][1] / (cm[0][0] + cm[0][1])
        LFR_1 = cm[1][0] / (cm[1][0] + cm[1][1])

        print(f"LFR for class 0: {LFR_0}")
        print(f"LFR for class 1: {LFR_1}")

    elif data_name == "AG":
        LFR_0 = (cm[0][1] + cm[0][2] + cm[0][3]) / (cm[0][0] + cm[0][1] + cm[0][2] + cm[0][3])
        LFR_1 = (cm[1][0] + cm[1][2] + cm[1][3]) / (cm[1][0] + cm[1][1] + cm[1][2] + cm[1][3])
        LFR_2 = (cm[2][0] + cm[2][1] + cm[2][3]) / (cm[2][0] + cm[2][1] + cm[2][2] + cm[2][3])
        LFR_3 = (cm[3][0] + cm[3][1] + cm[3][2]) / (cm[3][0] + cm[3][1] + cm[3][2] + cm[3][3])

        print(f"LFR for class 0: {LFR_0}")
        print(f"LFR for class 1: {LFR_1}")
        print(f"LFR for class 2: {LFR_2}")
        print(f"LFR for class 3: {LFR_3}")
        print(f"Average LFR:{(LFR_1 + LFR_2 + LFR_3) / 3}")

    else:
        raise ValueError("Invalid data name")

    return None
