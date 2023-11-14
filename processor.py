import torch
from gpt import GPT
from tokenizer import TinyTokenizer


class TextProcessor:
    def __init__(self, text):
        self.text = text

    def start(self):
        is_cuda = torch.cuda.is_available()
        device = "cuda:0" if is_cuda else "cpu"
        tk = (TinyTokenizer()).load()
        # org = "Little John John was a Jedi who trained in the dark arts."
        src = torch.tensor([tk.encode(self.text)]).to(device)
        trs = GPT().generate(src)
        # print(f"{org} - {tk.decode(trs.tolist()[0])}")
        return tk.decode(trs.tolist()[0])
