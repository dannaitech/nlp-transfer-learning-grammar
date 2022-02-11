"""
Train model defined in models.py using data in MSRDataset format.

Required Packages:
    pip install pyyaml
    pip install torch
    pip install transformers
    pip install sentencepiece

"""
import csv
import time
import datetime
import yaml

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW

from models import tokenizer, model
from MSRDataset import MSRDataset

config = None
with open('config.yaml') as fp:
    config = yaml.load(fp, Loader=yaml.FullLoader)

max_length = config['MAX_LEN']
batch_size = config['BATCH_SIZE']
epochs = config['EPOCHS']
lr = config['LEARNING_RATE']

model_savefile = config['CHECKPOINT_DIR']
tensorboard_logdir = config['LOG_DIR']

train_src_file = ''
train_tgt_file = ''
train_dataset = MSRDataset(train_src_file, train_tgt_file, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_src_file = ''
val_tgt_file = ''
val_dataset = MSRDataset(val_src_file, val_tgt_file, max_length)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

writer = SummaryWriter(tensorboard_logdir)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=lr)

train_start = time.time()

for epoch in range(epochs):
    epoch_start = time.time()

    for batch_idx, batch in enumerate(train_dataloader):
        model.train()
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device) 

        # Dropping last decoder input id to compensate for the
        # padding token added at the start to achieve right shift
        # This ensures equal tensor size of encoder and decoder inputs
        decoder_input_ids = batch['decoder_input_ids'][:, :-1].contiguous().to(device)
        # Copying all but the first decoder ids to be used as labels
        # because the first decoder id is the padding token id
        labels = batch['decoder_input_ids'][:, 1:].clone().to(device)
        # Converting all padding token ids to -100
        # since this id is ignored when calculating loss
        labels[labels[:, :] == tokenizer.pad_token_id] = -100

        """ Difference:-
            batch['decoder_input_ids'] = [0, 125, 1335, 1456, 0]    N x 5
            decoder_input_ids = [0, 125, 1335, 1456]                N x 4
            labels = [125, 1335, 1456, -100]                        N x 4

            Expected output = [125, 1335, 1456]
        """

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids = decoder_input_ids,
            labels = labels,
        )
        loss = outputs[0]
        writer.add_scalar('Training Loss', loss,
                          epoch * len(train_dataloader) + batch_idx)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_batch = next(iter(val_dataloader))

                val_input_ids = val_batch['input_ids'].to(device)
                val_attention_mask = val_batch['attention_mask'].to(device)

                val_decoder_input_ids = val_batch['decoder_input_ids'][:, :-1].contiguous().to(device)
                val_labels = val_batch['decoder_input_ids'][:, 1:].clone().to(device)
                val_labels[val_labels[:, :] == tokenizer.pad_token_id] = -100

                val_outputs = model(
                    input_ids=val_input_ids,
                    attention_mask=val_attention_mask,
                    decoder_input_ids = val_decoder_input_ids,
                    labels = val_labels
                )
                val_loss = val_outputs[0]
                writer.add_scalar('Validation Loss', val_loss,
                                  epoch * len(train_dataloader) + batch_idx)

                predicted_tokens = model.generate(
                    val_input_ids,
                    decoder_start_token_id=tokenizer.pad_token_id,
                    num_beams=5, early_stopping=True, max_length=max_length
                )
                print("INPUT: ", end='')
                print(tokenizer.batch_decode([val_input_ids[0]], skip_special_tokens=False)[0])
                print("TARGET: ", end='')
                print(tokenizer.batch_decode([val_decoder_input_ids[0]], skip_special_tokens=False)[0])
                print("PREDICT: ", end='')
                print(tokenizer.batch_decode([predicted_tokens[0]], skip_special_tokens=False)[0])

            writer.flush()

    epoch_end = time.time()
    print('Epoch Time: {:.1f} min'.format((epoch_end - epoch_start)/60))

writer.close()

train_end = time.time()
print('Total Training Time: {:.1f} min'.format((train_end - train_start)/60))

model.save_pretrained(model_savefile)
