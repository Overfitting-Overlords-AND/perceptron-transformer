from datasets import load_dataset
import tokenizer
import torch
import torch.utils.data


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.ds = load_dataset("roneneldan/TinyStories")
        self.tk = tokenizer.TinyTokenizer()
        self.tk.load()

    def __len__(self):
        return len(self.ds["train"])

    def __getitem__(self, idx):
        row = self.ds["train"][idx]["text"]
        input = [self.tk.sp.bos_id()] + self.tk.encode(row)
        label = (self.tk.encode(row)) + [self.tk.sp.eos_id()]
        return {"input": torch.tensor(input), "label": torch.tensor(label)}

    def collate_fn(self, batch):
        input_pad = torch.nn.utils.rnn.pad_sequence(
            [item["input"] for item in batch], batch_first=True, padding_value=0
        )
        label_pad = torch.nn.utils.rnn.pad_sequence(
            [item["label"] for item in batch], batch_first=True, padding_value=0
        )
        return {"input": input_pad, "label": label_pad}
