"""
Custom PyTorch Dataset class required for DataLoader

- Reads source and target sentences from csv file
- Uses MT5Tokenizer.prepare_seq2seq_batch() to
    - Convert tokens into ids
    - Generate attention masks

IMPORTANT: 'tgt_file' argument must be set as None when targets are not available
            e.g. for test data.

USAGE:
    dataset = MSRDataset('srcfile.csv', 'tgtfile.csv', 128)
    OR
    dataset = MSRDataset('srcfile.csv', None, 128)

Required Packages:
    pip install torch
    pip install transformers
    pip install sentencepiece
    
"""
import csv

from torch.utils.data import Dataset
from model import tokenizer


class MSRDataset(Dataset):
    def __init__(self, src_file, tgt_file, max_length):
        src = self.read_and_format(src_file, 'src')
        tgt = self.read_and_format(tgt_file, 'tgt')

        encodings = tokenizer.prepare_seq2seq_batch(
            src,
            tgt,
            padding='max_length',
            max_length=max_length,
            truncation=True,
            return_tensors='pt'
        )
        if tgt:
            self.data = {
                'input_ids': encodings.input_ids,
                'attention_mask': encodings.attention_mask,
                'decoder_input_ids': encodings.labels,
            }
        else:
             self.data = {
                'input_ids': encodings.input_ids,
                'attention_mask': encodings.attention_mask
            }          
        
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()} 

    def __len__(self):
        return len(self.data['input_ids'])

    def read_and_format(self, filename, format):
        """Reads sentences from csv file into a list

        This functions appends a 'pad' token to the start of target sentences to achieve
        the right shift required for decoder inputs in seq2seq models.

        Parameters
        ----------
        filename : str
            name of csv file e.g. sourcefile.csv
        format : ['src', 'tgt']
            Declare whether sentences are source (encoder inputs) or target (decoder inputs)

        Returns
        -------
        List of str
            List of sentences
        """
        sentences = None
        if filename:
            with open(filename, mode="r", encoding="utf-8", newline='') as fp:
                csv_reader = csv.reader(fp, delimiter = ' ')
                rows = list(csv_reader)

                if format == 'src':
                    sentences = [' '.join(row) for row in rows]
                elif format == 'tgt':
                    sentences = [tokenizer.pad_token + ' ' + ' '.join(row) for row in rows]
                else:
                    print("Error: Incorrect format specified")

        return sentences