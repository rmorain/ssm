from transformers import BertTokenizer, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
input_ids = tokenizer.encode("Hello, my dog is [MASK]", return_tensors="pt")
labels = tokenizer.encode("Hello, my dog is cute", return_tensors="pt")
outputs = model(input_ids, labels=labels)
loss, prediction_scores = outputs[:2]

tokens = torch.argmax(prediction_scores, dim=-1).squeeze()

print(tokenizer.decode(tokens))
import IPython ; IPython.embed() ; exit(1)