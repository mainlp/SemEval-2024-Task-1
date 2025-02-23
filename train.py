import os
import argparse
import pickle
import pandas as pd
from collections import defaultdict
from scipy.stats import spearmanr


import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding, TrainerCallback, EarlyStoppingCallback, AutoModelForSequenceClassification

import datasets


def get_args():
    parser = argparse.ArgumentParser(description='Train a LSTM classifier.')
    
    parser.add_argument('model_name', type=str, help='Training model name')
    parser.add_argument('save_path', type=str, help='Path to save results')
    parser.add_argument('train_file', type=str, help='Path to the training data file')
    parser.add_argument('pred_file', type=str, help='Path to results saving file')
    
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--train_batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=int, default=2e-5, help='Training learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Data and model seed')
    

    return parser.parse_args()


args = get_args()


save_path = os.path.abspath(args.save_path) 
pred_file = save_path + args.pred_file + '.pkl'
train_data = save_path + args.train_file + '.csv'

train_lang = args.train_file.split('/')[-1]
model_type = args.model_name.split('/')[-1]

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1).to(args.device)


def tokenize(example):
  return tokenizer(example["sent1"], example["sent2"], truncation=True, max_length=256)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    spearman_corr, _ = spearmanr(preds, labels)
    return {'spearman_corr': spearman_corr}


class MyCallback(TrainerCallback):
    """
    MyCallback
    """
    result = defaultdict(list)

    def on_epoch_end(self, args, state, control, **kwargs):
        print('MyCallback')
        print('Save predictions and labels after each epoch')
        
        
        with torch.no_grad():
            for i in range(0, len(tokenized_train_dataset), args.train_batch_size*8):
                batch_instances = tokenized_train_dataset[i:i+args.train_batch_size*8]

                # Extract data from the batch
                batch_instance_ids= batch_instances['PairID']
                batch_labels= batch_instances['labels']
                input_ids = batch_instances['input_ids']
                input_ids = pad_sequence([torch.tensor(seq) for seq in input_ids], batch_first=True, padding_value=1).to(args.device)

                attention_mask = batch_instances['attention_mask']
                attention_mask = pad_sequence([torch.tensor(seq) for seq in attention_mask], batch_first=True, padding_value=0).to(args.device)

                preds = model(input_ids, attention_mask).logits.detach().cpu().tolist()
                
                for pred, lbl, id in zip(preds, batch_labels, batch_instance_ids):
                    MyCallback.result[id].append((pred[0], lbl))


        with open(pred_file, 'wb') as file:
            pickle.dump(MyCallback.result, file)



# Load and split data
df = pd.read_csv(train_data)
train_dataset = datasets.Dataset.from_pandas(df)
train_dataset = train_dataset.shuffle(seed=args.seed)
tokenized_train_dataset = train_dataset.map(tokenize, batched=True, desc="Running tokenizer on train dataset")

train_set, val_set = torch.utils.data.random_split(tokenized_train_dataset, [0.9, 0.1])


training_args = TrainingArguments(
    output_dir=f'./logs/{model_type}_seed{args.seed}_{train_lang}',
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.train_batch_size*4,
    gradient_accumulation_steps=2,
    evaluation_strategy = "steps",
    eval_steps=200,
    learning_rate=args.lr,
    weight_decay=1e-3,
    num_train_epochs=args.num_epochs,
    fp16=True if torch.cuda.is_available() else False,
    logging_strategy="steps",
    logging_steps=len(train_set)//args.train_batch_size,
    logging_dir='./logs/' + model_type,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=1,
    seed=args.seed,
    data_seed=args.seed,
    dataloader_num_workers=2,
    metric_for_best_model='loss',
    load_best_model_at_end="True",
    report_to=None
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=val_set,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    callbacks=[MyCallback, EarlyStoppingCallback(early_stopping_patience=8, early_stopping_threshold=0.0001)]
)


trainer.train()
trainer.push_to_hub()