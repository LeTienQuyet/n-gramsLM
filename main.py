from collections import defaultdict, Counter

import pickle
import os
import re

def prepare_data(dir_data: str="data.txt") -> list[str]:
    with open(dir_data, 'r', encoding="utf-8") as file:
        lines = file.readlines()
    data = [line.strip() for line in lines]
    return data
def preprocess_data(data: list[str]) -> list[list[str]]:
    return [[word.lower() for word in re.sub(r'[^\w\s]', '', sentence).split()] for sentence in data]

def ngram_model(data_tokenized: list[list[str]], ngram: int=2) -> defaultdict:
    counting = defaultdict(Counter)
    for tokens in data_tokenized:
        for i in range(len(tokens)-ngram+1):
            prefix = tuple(tokens[i:i+ngram-1])
            next_word = tokens[i+ngram-1]
            counting[prefix][next_word] += 1
    return counting

def prod_model(counting: defaultdict) -> dict:
    model = {}
    for prefix, next_word in counting.items():
        total_count = sum(next_word.values())
        model[prefix] = {word: count / total_count for word, count in next_word.items()}
    return model

def save_model(model: dict, model_name: str, dir_checkpoint: str="checkpoint"):
    os.makedirs(dir_checkpoint, exist_ok=True)
    with open(os.path.join(dir_checkpoint, model_name), "wb") as file:
        pickle.dump(model, file)
    print("Save model successful !!!")

def main(data_tokenzied: list[list[str]], ngram: int=2):
    counting = ngram_model(data_tokenzied, ngram)
    model = prod_model(counting)
    save_model(model, f"{ngram}grams.pkl")

if __name__ == "__main__":
    data = prepare_data()
    data_tokenzied = preprocess_data(data)
    for ngram in range(2, 6):
        main(data_tokenzied, ngram)
    print("Done !!!")

