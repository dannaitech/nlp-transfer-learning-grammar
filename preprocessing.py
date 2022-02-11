"""
The dataset consists of Universal Dependency structures in conllu format.
We extract sentences from these structures and discard other information.

Required Packages:
    pip install conllu

USAGE:
    e.g. filename is 'en_ewt-ud-train.conllu'
    extract_sentences_from_conllu_to_csv('src/', 'en_ewt-ud-train', 'dst/')
    
"""

import csv
from conllu import parse


def extract_sentences_from_conllu_to_csv(src_dir, name, dst_dir):
    """Extract sentences from Universal Dependency structures in .conllu file
    and store in .csv file.

    Parameters
    ----------
    src_dir : str
        directory of .conllu file
    name : str
        name of .conllu file excluding '-train.conllu' or '-test.conllu'
    mode : str
        type of conllu file, 'train', 'test' etc.
    dst_dir : str
        directory where .csv file should be saved
    """
    filename = src_dir + name + '.conllu'
    print('Processing ' + filename + ' ... ', end='', flush=True)

    raw_data = open(filename, "r", encoding="utf-8").read()
    ud_dataset = parse(raw_data)

    sentences = []
    for tokenlist in ud_dataset:
        sentence = []
        for token in tokenlist:
            word = token['form']
            sentence.append(word)
        sentences.append(sentence)

    filename = dst_dir + name + '.csv'
    with open(filename, mode="w", encoding="utf-8", newline='') as fp:
        csv_writer = csv.writer(fp, delimiter=' ')
        csv_writer.writerows(sentences)

    print('DONE ', end='')
    print((len(ud_dataset), len(sentences)))
