import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="ary_Arab")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", token=True).to(device)
model.eval()


def translate_batch(path, source, target, batch_size=32):
    language_mapping = {"amh": "amh_Ethi", "ary": "ary_Arab", "eng": "eng_Latn", "esp": "spa_Latn", "hau": "hau_Latn", "kin": "kin_Latn", "mar": "mar_Deva", "tel": "tel_Telu"}
    original_csv_file_path = path + f'{source}_train.csv'
    df = pd.read_csv(original_csv_file_path).dropna()

    tokenizer.src_lang = language_mapping[source]

    translated_sent1 = []
    translated_sent2 = []

    for i in tqdm(range(0, len(df), batch_size), unit="batches", desc='Translating...'):
        batch_df = df.iloc[i:i+batch_size]

        # Generate translation for sent1
        sent1_tokenized = tokenizer(batch_df['sent1'].tolist(), return_tensors='pt', padding=True, truncation=True).to(device)
        sent1_generated = model.generate(**sent1_tokenized, forced_bos_token_id=tokenizer.lang_code_to_id[language_mapping[target]])
        sent1_translated = tokenizer.batch_decode(sent1_generated, skip_special_tokens=True)
        translated_sent1.extend(sent1_translated)

        # Generate translation for sent2
        sent2_tokenized = tokenizer(batch_df['sent2'].tolist(), return_tensors='pt', padding=True, truncation=True).to(device)
        sent2_generated = model.generate(**sent2_tokenized, forced_bos_token_id=tokenizer.lang_code_to_id[language_mapping[target]])
        sent2_translated = tokenizer.batch_decode(sent2_generated, skip_special_tokens=True)
        translated_sent2.extend(sent2_translated)

    df['sent1'] = translated_sent1
    df['sent2'] = translated_sent2
    df['PairID'] = f"{source.upper()}-{target.upper()}-train-" + df.index.astype(str).str.zfill(4)

    translated_csv_file_path = path + f'{source}-{target}_train.csv'
    df.to_csv(translated_csv_file_path, index=False)

    return df