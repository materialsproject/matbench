# model_type model_name
# llama baffo32/decapoda-research-llama-7B-hf
# llama NousResearch/Llama-2-7b-hf
# auto Undi95/Meta-Llama-3-8B-hf
# auto "bert-base-uncased"
# auto "bert-base-cased"
# auto "bert-large-cased"
import gc
import os
os.environ['TRANSFORMERS_CACHE'] = '/data/cache/'
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import fire
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import PowerTransformer
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoModel, AutoTokenizer
import torch, gzip, json
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import lightning as L
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from constants import units_dict, des_dict, task_type_dict

pt = PowerTransformer(method='yeo-johnson')

class LLMRegressor(L.LightningModule):
    def __init__(self, model_type, model_name, opt_llm=False, load_in_8=False, reg_or_cls="regression", trans=False):
        super(LLMRegressor, self).__init__()
        self.opt_llm = opt_llm
        self.reg_or_cls = reg_or_cls
        self.trans=trans
        self.model_type = model_type
        if model_type == "llama":
            self.model = LlamaForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=load_in_8,
            torch_dtype=torch.float32,
            device_map="auto",
            cache_dir='/data/cache',
            output_hidden_states=True
        )
        elif model_type == "auto":
            self.model = AutoModel.from_pretrained(model_name, cache_dir='/data/cache', torch_dtype=torch.float32, load_in_8bit=load_in_8)
        else:
            pass

        # Freeze the LLM model
        for param in self.model.parameters():
            if param.dtype.is_floating_point or param.dtype.is_complex:
                param.requires_grad = opt_llm

        self.regression_head = torch.nn.Linear(self.model.config.hidden_size, 1)# .to(dtype=torch.float16)
        if self.reg_or_cls == "cls_reg_both":
            self.extra_cls_head = torch.nn.Linear(self.model.config.hidden_size, 1)# .to(dtype=torch.float16)
        self.predictions = []
        self.ratings = []

    def forward(self, texts, attention_mask):
        # inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        # inputs = {k: v for k, v in inputs.items()}
        outputs = self.model(texts, attention_mask=attention_mask)
        if self.model_type == "auto":
            cls_representation = outputs.last_hidden_state[:, 0, :]
        elif self.model_type == "llama":
            all_hidden_states = outputs.hidden_states
            cls_representation = all_hidden_states[-1][:, 0, :]  # 获取最后一层的所有CLS标记的输出
        else:
            pass
        regression_value = self.regression_head(cls_representation).squeeze(-1)
        extra_cls_value = None
        if self.reg_or_cls == "classification":
            regression_value = torch.sigmoid(regression_value)
        elif self.reg_or_cls == "cls_reg_both":
            extra_cls_value = torch.sigmoid(self.extra_cls_head(cls_representation).squeeze(-1))
        return extra_cls_value, regression_value

    def training_step(self, batch, batch_idx):
        texts, mask, ratings = batch
        texts = texts.to(self.device)  # Move texts to GPU
        ratings = ratings.to(self.device)  # Move ratings to GPU
        mask = mask.to(self.device)  # Move ratings to GPU
        extra, predictions = self(texts, mask)
        if self.reg_or_cls == "regression":
            loss = torch.nn.functional.mse_loss(predictions, ratings)
        elif self.reg_or_cls == "cls_reg_both":
            classification_loss = F.binary_cross_entropy(extra, (ratings > 0).float())
            class_output_rounded = torch.round(extra)
            reg_mask = class_output_rounded < 0.5
            predictions[reg_mask] = 0
            regression_loss = torch.nn.functional.mse_loss(predictions, ratings)
            loss = classification_loss + regression_loss
        else:
            loss = F.binary_cross_entropy(predictions, ratings.float())
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        texts, mask, ratings = batch
        texts = texts.to(self.device)  # Move texts to GPU
        ratings = ratings.to(self.device)  # Move ratings to GPU
        mask = mask.to(self.device)  # Move ratings to GPU
        extra, predictions = self(texts, mask)
        if self.reg_or_cls == "regression":
            loss = torch.nn.functional.mse_loss(predictions, ratings)
        elif self.reg_or_cls == "cls_reg_both":
            class_output_rounded = torch.round(extra)
            reg_mask = class_output_rounded < 0.5
            predictions[reg_mask] = 0
            regression_loss = torch.nn.functional.mse_loss(predictions, ratings)
            loss = regression_loss
        else:
            loss = F.binary_cross_entropy(predictions, ratings.float())

        self.predictions.extend(predictions)
        self.ratings.extend(ratings)
        return loss

    def on_validation_epoch_end(self):
        preds = torch.tensor(self.predictions)
        ratings = torch.tensor(self.ratings)
        if self.reg_or_cls == "classification":
            loss = F.binary_cross_entropy(preds, ratings.float())
        else:
            loss = torch.nn.functional.l1_loss(preds, ratings)
        self.log('val_loss', loss)
        self.predictions = []
        self.ratings = []

    def test_step(self, batch, batch_idx):
        texts, mask, ratings = batch
        extra, predictions = self(texts, mask)
        if self.trans:
            predictions = pt.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).flatten()
            predictions = torch.tensor(predictions, dtype=torch.float32, device=ratings.device)
        if self.reg_or_cls == "cls_reg_both":
            class_output_rounded = torch.round(extra)
            reg_mask = class_output_rounded < 0.5
            predictions[reg_mask] = 0

        loss = torch.nn.functional.mse_loss(predictions, ratings)
        self.predictions.extend(predictions)
        self.ratings.extend(ratings)
        return {'predictions': predictions, 'ratings': ratings}

    def on_test_epoch_end(self):
        preds = torch.tensor(self.predictions)
        ratings = torch.tensor(self.ratings)

        # 存储到 CSV 文件
        preds_np = preds.cpu().numpy()  # 转换为 NumPy 数组
        ratings_np = ratings.cpu().numpy()  # 转换为 NumPy 数组

        # 创建 DataFrame
        df = pd.DataFrame({
            'predictions': preds_np,
            'ratings': ratings_np
        })

        if self.reg_or_cls == "classification":
            mae = roc_auc_score(ratings.cpu().numpy(), torch.round(preds).cpu().numpy())
        else:
            mae = torch.nn.functional.l1_loss(preds, ratings)
        self.log('test_mae', mae)
        
        log_dir = self.logger.log_dir if hasattr(self, 'logger') and self.logger else './logs'
        file_path = os.path.join(log_dir, f'predictions_mae_{round(float(mae), 4)}.csv')
        df.to_csv(file_path, index=False)

    def configure_optimizers(self):
        if self.opt_llm:
            l_r = 1e-5
        else:
            l_r = 1e-4
        optimizer = AdamW(self.parameters(), lr=l_r)
        return optimizer


class QADataset(Dataset):
    def __init__(self, data_json, tokenizer, max_length=512, trans=False):
        # self.texts = [f"Q: {q} A: {a}" for q, a in zip(questions, answers)]
        # self.ratings = torch.tensor(ratings, dtype=torch.float32)
        
        self.texts = [f"{dj['instruction']} {dj['input']}" for dj in data_json]
        ratings = [1.0 if dj['output'] == 'True' else 
                    0.0 if dj['output'] == 'False' else
                    float(dj['output']) for dj in data_json]
        if trans:
            ratings = np.array(ratings)
            ratings = pt.fit_transform(ratings.reshape(-1, 1)).flatten()
        self.ratings = torch.tensor(ratings, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # print(self.texts[idx], self.ratings[idx])
        # Tokenize the text input for the model

        # replace number with [unusedxxx], only int???

        encoded_pair = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # Extract the input ids and attention mask from the encoding
        input_ids = encoded_pair['input_ids'].squeeze(0)  # Squeeze is used to remove the batch dimension
        attention_mask = encoded_pair['attention_mask'].squeeze(0)

        return input_ids, attention_mask, self.ratings[idx]


def main(
    model_type: str="auto",
    model_name: str="bert-base-uncased",
    opt_llm: bool=False,
    fold: int=0,
    task_name: str="matbench_jdft2d",
    train_cnt: int=0,
    test_cnt: int=0,
    pretrained_ckpt_path: str="",
        ):
    print(model_type, model_name)
    # Sample data
    # questions = ["How is the plot?", "How is the acting?", "How are the effects?"]
    # answers = ["It's good.", "It's okay.", "It's excellent."]
    # ratings = [7.5, 6.0, 8.5]
    if model_type == "llama":
        tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir='/data/cache') # , local_files_only=True)
    elif model_type == "auto":
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='/data/cache')
    else:
        pass
    l8 = False
    if "lama" in model_name:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        l8 = True
    folder_name = "nl"
    with gzip.open(f'data_{folder_name}_v1/Train_{fold}_{task_name}_{folder_name}_{train_cnt}.json.gz', 'rb') as f:
        train_decompressed_data = f.read().decode('utf-8')
    
    train_data_loaded = json.loads(train_decompressed_data)
    train_cnt = int(0.9*len(train_data_loaded))
    vali_cnt = len(train_data_loaded) - train_cnt
    
    trans = task_type_dict[task_name][2] # yeo-johnson transformation
    
    train_dataset, vali_dataset = random_split(QADataset(train_data_loaded, tokenizer, trans=trans), [train_cnt, vali_cnt])
    
    with gzip.open(f'data_{folder_name}_v1/Test_{fold}_{task_name}_{folder_name}_{test_cnt}.json.gz', 'rb') as f:
        test_decompressed_data = f.read().decode('utf-8')
    
    test_data_loaded = json.loads(test_decompressed_data)
    test_dataset = QADataset(test_data_loaded, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    vali_loader = DataLoader(vali_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize model and training components
    model = LLMRegressor(model_type, model_name, opt_llm, l8, reg_or_cls=task_type_dict[task_name][1], trans=trans)
    logger = TensorBoardLogger("logs_nl_best_0.1vali_200patience_with_mask", name=f"{model_name}/{task_name}_{fold}_{opt_llm}")
    m_e = 5000
    checkpoint_callback = ModelCheckpoint(monitor="val_loss",
                                         mode="min",
                                         save_weights_only=True,
                                         save_top_k=1,
                                         filename='{epoch}-{val_loss:.4f}')
    earlystopping_callback = EarlyStopping(
                                     monitor="val_loss",
                                     mode="min",
                                     patience=200)
    trainer = L.Trainer(max_epochs=m_e, logger=logger, callbacks=[checkpoint_callback, earlystopping_callback])

    # Training loop
    if pretrained_ckpt_path == "":
        trainer.fit(model, train_loader, vali_loader)
        best_model_path = checkpoint_callback.best_model_path

        del model.model, checkpoint_callback, train_loader, vali_loader, earlystopping_callback, model
        torch.cuda.empty_cache()
        gc.collect()
        checkpoint = torch.load(best_model_path)
        model = LLMRegressor(model_type, model_name, opt_llm, l8, reg_or_cls=task_type_dict[task_name][1], trans=trans).cuda()
    elif pretrained_ckpt_path == "originBERT":
        model = LLMRegressor(model_type, model_name, opt_llm, l8, reg_or_cls=task_type_dict[task_name][1], trans=trans)
    else:
        folder_path = f"{pretrained_ckpt_path}/{model_name}/{task_name}_{fold}_{opt_llm}/version_0/checkpoints/"
        ckpt_path = glob.glob(os.path.join(folder_path, "*.ckpt"))
        if len(ckpt_path) == 1:
            print(f"Loading checkpoint: {ckpt_path[0]}")
            checkpoint = torch.load(ckpt_path[0], map_location=torch.device('cpu'))
        elif len(ckpt_path) == 0:
            raise FileNotFoundError(f"No .ckpt file found in folder: {folder_path}")
        else:
            raise ValueError(f"Multiple .ckpt files found in folder: {folder_path}")
        model = LLMRegressor(model_type, model_name, opt_llm, l8, reg_or_cls=task_type_dict[task_name][1], trans=trans)

    if pretrained_ckpt_path != "originBERT":
        model_state_dict = checkpoint['state_dict']
        model.load_state_dict(model_state_dict)
        model.eval()

    trainer.test(model, test_loader)

if __name__ == "__main__":
    fire.Fire(main)
