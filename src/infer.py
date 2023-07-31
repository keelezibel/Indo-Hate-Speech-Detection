import torch
import torch.nn.functional as F
from fastapi import FastAPI
from transformers import BertTokenizer
from multi_label_classification import BertForMultiLabelClassification
from pydantic import BaseModel
from transformers import AutoConfig

HS_DOMAIN = ['hs', 'abusive', 'hs_individual', 'hs_group', 'hs_religion', 'hs_race', 'hs_physical', 'hs_gender', 'hs_other', 'hs_weak', 'hs_moderate', 'hs_strong']
LABEL2INDEX = {'false': 0, 'true': 1}
INDEX2LABEL = {0: 'false', 1: 'true'}

class Query(BaseModel):
    text: str

app = FastAPI()

def load_model():
    # Load model
    tokenizer_model_id = "indobenchmark/indobert-base-p2"
    tokenizer = BertTokenizer.from_pretrained(tokenizer_model_id)
    config = AutoConfig.from_pretrained(tokenizer_model_id)
    config.num_labels_list = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    model_id = "keelezibel/id-hatespeech"
    model = BertForMultiLabelClassification.from_pretrained(model_id, config=config)
    return tokenizer, model
    
@app.post("/infer")
async def infer(text: Query):
    tokenizer, model = load_model()
    subwords = tokenizer.encode(text.text)
    subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)

    logits = model(subwords)[0]
    labels = [torch.topk(logit, k=1, dim=-1)[-1].squeeze().item() for logit in logits]

    res = dict()
    for idx, label in enumerate(labels):
        pred = INDEX2LABEL[label]
        proba = float(F.softmax(logits[idx], dim=-1).squeeze()[label]*100)
        res[HS_DOMAIN[idx]] = (pred, round(proba,2))
    return res
