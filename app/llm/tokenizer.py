import numpy as np

class SimpleCharTokenizer:
    def __init__(self):
       
        chars = list("abcdefghijklmnopqrstuvwxyz0123456789 .,!?;:'\"()[]{}-_/\\@#$%^&*\n")
        
        
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3

       
        self.vocab = ["<pad>", "<bos>", "<eos>", "<unk>"] + chars


       
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

   
    @property
    def vocab_size(self):
        return len(self.vocab)

 
    def encode(self, text, add_special_tokens=True):
        text = text.lower()
        ids = [self.stoi.get(ch, self.unk_id) for ch in text]
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return np.array(ids, dtype=np.int32)

   
    def decode(self, ids):
        out = []
        for i in ids:
            if i in (self.pad_id, self.bos_id): 
                continue
            if i == self.eos_id:  
                break
            out.append(self.itos.get(int(i), ""))
        return "".join(out)
