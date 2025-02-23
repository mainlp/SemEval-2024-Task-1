import os
from uroman import Uroman
import pandas as pd


def create_transliteration_data(file_path):

    directory, filename = os.path.split(file_path)
    print(filename)

    assert os.path.isfile(file_path), f"Input file path {file_path} not found"
    if (os.path.exists(f"{directory}/text_transliterations.csv") and "without" in filename) or \
    (os.path.exists(f"{directory}/text_transliterations_with_latn.csv") and "without" not in filename):
        pass
    else:
        roman = Uroman()
        df = pd.read_csv(file_path)
        sent1_list = df['sent1'].tolist()
        sent2_list = df['sent2'].tolist()

        sent1_transliterations = roman.romanize(sent1_list, './temp')
        sent2_transliterations = roman.romanize(sent2_list, './temp')
        assert len(sent1_list) == len(sent1_transliterations)
        assert len(sent2_list) == len(sent2_transliterations)

        # write transliterations:
        examples = []
        df['sent1_latin'] = sent1_transliterations
        df['sent2_latin'] = sent2_transliterations

        df.to_csv(file_path, index=False)

create_transliteration_data("dev_dataset/amh_dev_with_labels.csv")
create_transliteration_data("dev_dataset/arq_dev_with_labels.csv")
create_transliteration_data("dev_dataset/ary_dev_with_labels.csv")
create_transliteration_data("dev_dataset/mar_dev_with_labels.csv")
create_transliteration_data("dev_dataset/tel_dev_with_labels.csv")
