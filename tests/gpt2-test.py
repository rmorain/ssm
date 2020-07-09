from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained('gpt2')

generated = tokenizer.encode("What year did Lord of the Rings\
     come out? Lord of the Rings released year 2001")
context = torch.tensor([generated])
past = None

for i in range(100):
    output, past = model(context, past=past)
    probs = output[..., -1, :]
    m = torch.distributions.Categorical(probs)
    token = m.sample()
    generated += [token[0].tolist()]
    context = token.view(1, -1)

sequence = tokenizer.decode(generated)
print(sequence)