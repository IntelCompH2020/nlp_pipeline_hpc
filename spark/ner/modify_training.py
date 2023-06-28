import os
import random

def read_sentences(filename):
    """
    Read conll format into sentences (separated by one read line).
    """
    lines = None 
    with open(filename) as f:
        lines = f.readlines()
    sentences = []
    sentence = []
    for line in lines:
        if len(line) == 1 and line == '\n':
            sentences.append(sentence)
            sentence = []
        else:
            text, label = line.split(' ', 1)
            sentence.append([text, label])
    if len(sentence) > 0:
        sentences.append(sentence)
        sentence = []
    return sentences


def apply_to_sentences(sentences, function=lambda x: x.lower(), percentage=1.0):
    """
    Given a list of sentences, the given function will be applied 
    to each word of some sentences, the sentences will be randomly 
    selected with a given probability (float between 0 and 1).
    """
    modified_sentences = []
    for sentence in sentences:
        if random.random() < percentage:
            new_sentence = []
            for text, label in sentence:
                if text != "-DOCSTART-":
                    new_text = function(text)
                else:
                    new_text = text
                new_sentence.append([new_text, label])
            modified_sentences.append(new_sentence)
        else:
            modified_sentences.append(sentence)
    return modified_sentences


def write_sentences(sentences, filename):
    """
    Given a list of sentences (lists of text-label pairs), 
    write them in filename in conll format.
    """
    with open(filename, 'w') as f:
        for sentence in sentences:
            for text, label in sentence:
                f.write(text + ' ' + label)
            f.write('\n')


if __name__ == "__main__":
    dataset_splits = ["esp.train"]
    for dataset_split in dataset_splits:
        sentences = read_sentences(dataset_split)
        lower_sentences = apply_to_sentences(sentences, function=lambda x: x.lower(), percentage=1.0)
        upper_sentences = apply_to_sentences(sentences, function=lambda x: x.upper(), percentage=1.0)
        joined_sentences = sentences + lower_sentences + upper_sentences
        write_sentences(joined_sentences, 'joined_' + dataset_split)