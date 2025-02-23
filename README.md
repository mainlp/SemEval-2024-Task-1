## Files

1. `lang_sim.py`: This script calculates language similarity using cosine similarity between language vectors sourced from the URIEL database.

2. `nlln_translate.py`: This script handles translation tasks using the NLLB, which supports high-quality translations between a wide range of low-resource languages.

3. `preprocess_dataset.py`: This script provides functionality for transliteration tasks, converting text from one script to another.

4. `train.py`: This script is used for model training.


## Example Usage

Model Training: python3 train.py {model_name} {save_path} {train_file} {pred_file}

## You can find the following fine-tuned models on Huggingface:

Shijia/furina_seed42_eng
Shijia/furina_seed42_eng_amh_esp
Shijia/furina_seed42_eng_kin_amh
Shijia/furina_seed42_eng_amh_hau
Shijia/furina_seed42_eng_kin_hau
Shijia/furina_seed42_eng_esp_hau
Shijia/furina_seed42_eng_esp_kin
Shijia/furina_seed42_eng_amh_hau_roman
Shijia/furina_seed42_eng_kin_amh_roman
Shijia/furina_seed42_eng_amh_esp_roman

shanhy/xlm-roberta-base_seed42_eng_train
shanhy/xlm-roberta-base_seed42_esp-kin-eng_train
shanhy/xlm-roberta-base_seed42_kin-hau-eng_train
shanhy/xlm-roberta-base_seed42_amh-hau-eng_train
shanhy/xlm-roberta-base_seed42_kin-amh-eng_train
shanhy/xlm-roberta-base_seed42_amh-esp-eng_train
shanhy/xlm-roberta-base_seed42_esp-hau-eng_train
