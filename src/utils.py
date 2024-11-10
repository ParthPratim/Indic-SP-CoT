# from api import APIModel, pack_message
import argparse
import datetime
import json
import multiprocessing
import os
import random
import re
from statistics import mean
from string import punctuation, whitespace, ascii_lowercase
from typing import List, Tuple, Dict
import nltk
from inltk.inltk import setup 
setup('hi') 
from inltk.inltk import tokenize
import numpy as np
import spacy
import torch
from nltk.corpus import stopwords
from torch.utils.data import Dataset
from collections import Counter
import unidecode
from tqdm import tqdm

# lemmatizer = nlp.get_pipe("lemmatizer")


expanded_punc_and_whitespace = punctuation + whitespace + "-"

ban_word = ['हूँ', 'है', 'हैं', 'होना', 'था', 'थे', 'था', 'थी', 'हो', 'किया', 'जो', 'किसे', 'क्या', 'कब', 'क्यों', 'कैसे', 'करना', 'किया', 'लगभग', 'अधिकांश', 'था', '\'t', '\'s', 'हूँ', 'है', 'हैं', 'होना', 'था', 'थे', 'था', 'थी', 'हो', 'किया', 'जो', 'किसे', 'क्या', 'कब', 'क्यों', 'कैसे', 'करना', 'किया', 'लगभग', 'अधिकांश', 'था']

MONTHS = ['जनवरी', 'फरवरी', 'मार्च', 'अप्रैल', 'मई', 'जून', 'जुलाई', 'अगस्त', 'सितंबर', 'अक्टूबर', 'नवंबर', 'दिसंबर']

DAYS = ['सोमवार', 'मंगलवार', 'बुधवार', 'गुरुवार', 'शुक्रवार', 'शनिवार', 'रविवार']

DATE_PATTERN = r'(जन|फर|मार्च|अप्रैल|मई|जून|जुलाई|अग|सित|अक्ट|नव|दिस)\w*[, ]{1}\d{4}'

DEFAULT_PATHS = {
    "hotpot-qa": "data/hotpot-qa",
    "cweb-qa": "data/cweb-qa",
    "musique-qa": "data/musique-qa",
    "wikimh-qa": "data/wikimh-qa",
    "self-prompt": "data/self-prompt",
    "self-prompt-cot": "data/self-prompt-cot",
    "grail-qa": "data/grail-qa",
}

TYPE_MAPPING = {
    "type_1": 0,
    "type_2": 1,
    "type_3": 2,
    "type_4": 3,
    "type_5": 4,
    "type_6": 5,
}

DEFAULT_TRAIN_PATHS = {
    "hotpot-qa": "data/hotpot-qa/hotpot_train_v1.1.json",
    "cweb-qa": "data/cweb-qa/train.json",
    "musique-qa": "data/musique-qa/musique_ans_v1.0_train.jsonl",
    "wikimh-qa": "data/wikimh-qa/train.json",
    "grail-qa": "data/grail-qa/train.json",
}


DEFAULT_DEV_PATHS = {
    "hotpot-qa": "data/hotpot-qa/hotpot_dev_distractor_v1.json",
    "cweb-qa": "data/cweb-qa/dev.json",
    "musique-qa": "data/musique-qa/musique_ans_v1.0_dev.jsonl",
    "wikimh-qa": "data/wikimh-qa/dev.json",
    "grail-qa": "data/grail-qa/dev.json",
}

DEFAULT_TEST_PATHS = {
    "hotpot-qa": "data/hotpot-qa/hotpot_test_fullwiki_v1.json",
    "cweb-qa": "data/cweb-qa/test.json",
    "musique-qa": "data/musique-qa/musique_ans_v1.0_test.jsonl",
    "wikimh-qa": "data/wikimh-qa/test.json",
    "grail-qa": "data/grail-qa/test.json",
}

stop_set = set([
    "के", "का", "की", "से", "है", "हैं", "को", "में", "कि", "यह", "और", "इस", 
    "पर", "इससे", "था", "थे", "तो", "भी", "नहीं", "हो", "जो", "कर", "लेकिन", 
    "अपने", "द्वारा", "किया", "होता", "हुआ", "तक", "सकता", "रहा", "इसी", 
    "इसे", "थी", "दो", "रहे", "करना", "सबसे", "तरह", "करके", "करे", "वह", 
    "वे", "मगर", "कौन", "किस", "जैसे", "सभी", "व", "अब", "जब", "या", "यदि", 
    "जैसा", "कुछ", "दूसरे", "अधिक", "कहीं", "बनी", "आप", "फिर", "थोड़ा", "अभी", 
    "जाते", "उनके", "इन", "उन", "बाद", "इसका", "इन्हें", "इन्हों", "चुका", 
    "इत्यादि", "जाता", "जाती", "जाते", "तुम", "आपको", "तब", "इसमें", "रखें", 
    "का", "थे", "वर्ग", "कई", "थी", "गए", "उनका", "इन्हें", "इनका", "कुछ", 
    "कुल", "एस", "रहा", "जिस", "एस", "जिन्हें", "तुम्हारा", "कैसे", "सकते", 
    "सब", "दिया", "जाता", "उस", "अत", "पहले", "बनी", "इनका", "जिसे", "जाना", 
    "अंदर", "उनको", "इससे", "लिए", "अपने", "साथ", "आज", "तक", "अधिकांश", "काफी", 
    "द्वारा", "इसके", "उसके", "कुछ", "कहा", "हर", "जिन्होंने", "वह", "वही", 
    "जिन", "जितना", "सकता", "करें", "उस", "उन", "नीचे", "बिना", "होती", "उसका", 
    "सकती", "आज", "उन्हें", "उसके", "किसी", "होते", "बताया", "चाहिए", "मैं", 
    "मेरा", "हम", "हमारा", "मुझे", "हमने", "मेरे", "हमारे"
])



def check_answer(text):
    pattern = r"^(हाँ|नहीं)[,.]?"
    match = re.match(pattern, text, re.IGNORECASE)
    if match:
        answer = match.group(0).lower()  # Get the matched string in lowercase
        answer = re.sub(r'[,.]', '', answer)  # Remove punctuation
        return answer
    else:
        return None

import stanza

stanza.download('hi')
hnlp = stanza.Pipeline('hi')

def lemmatize(text):
    
    # Process the text
    doc = hnlp(text)
    
    # Extract lemmas
    lemmas = []
    for sent in doc.sentences:
        for word in sent.words:
            lemmas.append(word.lemma if word.lemma else word.text)
    
    return " ".join(lemmas)


def process_an_item(sentence):
    # Define patterns and stopwords
    quote_pattern = r"‘.*?’|“.*?”|(?='.*?')(?=(?!'t\s))(?=(?!'s\s))(?=(?!'ve\s))(?=(?!'m\s))|\".*?\""
    continuous_cap_words_pattern = r"([A-Z][^\s\?]*(?=\s[A-Z])(?:\s[A-Z][^\s\?]*)*)"

    # Extract quotes and replace them with a placeholder
    quotes = re.findall(quote_pattern, sentence)
    sentence = re.sub(quote_pattern, "' '", sentence)

    # Extract and refine continuous capitalized words, and replace them with a placeholder
    cont_cap_words = re.findall(continuous_cap_words_pattern, sentence)
    refined_ones = [item for item in cont_cap_words if not any(item.startswith(banword + ' ') for banword in ban_word)]
    sentence = re.sub(continuous_cap_words_pattern, "' '", sentence)

    # Tokenize, POS-tag, and extract named entities
    tokens = tokenize(sentence, 'hi')
    postags = nltk.pos_tag(tokens)
    named_entities = nltk.ne_chunk(postags, binary=False)

    # Collect named entities
    ss = []
    for t in named_entities:
        if hasattr(t, "label"):
            b = " ".join([x[0] for x in t])
        else:
            continue
        if b not in stop_set and b not in punctuation and b not in ss:
            ss.append(b)

    return quotes + refined_ones + ss


def remove_punc(text):
    exclude = set(punctuation)
    return "".join(ch for ch in text if ch not in exclude)


def include_without_punc(text1: str, text2: str, min_length: int = 4):
    text1 = remove_punc(text1).lower()
    if len(text1) < min_length:
        return False
    text2 = remove_punc(text2).lower()

    # find the next char of text2 after text1 if text1 in text2
    if text1 in text2:
        idx = text2.find(text1)
        if idx + len(text1) < len(text2):
            next_char = text2[idx + len(text1)]
            if next_char == "s":
                return True
            elif next_char in ascii_lowercase:
                return False
            return True
        else:
            return False
    return False


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(एक|यह|वह|इस|उस)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def lower(text):
        return text.lower()

    s = unidecode.unidecode(s)
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def shuffle_dict(d):
    keys = list(d.keys())
    random.shuffle(keys)
    keys = [(key, d[key]) for key in keys]
    random.shuffle(keys)
    keys = [(key, d[key]) for key in keys]
    random.shuffle(keys)
    keys = [(key, d[key]) for key in keys]
    # keys = d(keys)
    return dict(keys)


def fix_seed(seed: int = 42):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def print_now(return_flag=0):
    t_delta = datetime.timedelta(hours=8)
    CST = datetime.timezone(t_delta, 'CST')
    now = datetime.datetime.now(CST)
    now = now.strftime('%Y/%m/%d %H:%M:%S')
    if return_flag == 0:
        print(now)
    elif return_flag == 1:
        return now
    else:
        pass


LABEL_MAPPING = {
    "REFUTES": "false",
    "SUPPORTS": "true",
    "NOT_SUPPORTED": "false",
    "SUPPORTED": "true",
    "true": "true",
    "True": "true",
    "false": "false",
    "False": "false",
    "yes": "true",
    "Yes": "true",
    "no": "false",
    "No": "false",
}


def load_json_and_jsonl_file(input_file: str):
    if input_file.endswith(".json"):
        with open(input_file, "r") as f:
            raw_dataset = json.load(f)
    elif input_file.endswith(".jsonl"):
        raw_dataset = []
        with open(input_file, "r") as f:
            for line in f:
                raw_dataset.append(json.loads(line))
    else:
        raise ValueError(f"File format {input_file} is not supported.")
    return raw_dataset


def data_reader(dataset_path):
    q_len_list = []
    a_len_list = []
    n_steps_list = []

    raw_dataset = load_json_and_jsonl_file(dataset_path)
    for item in tqdm(raw_dataset):
        if "question" in item:
            q_len_list.append(len(item["question"].split(" ")))
        else:
            raise ValueError(f"Question is not found in {item}.")

        if "answer" in item:
            raw_answers = item["answer"]
            if isinstance(raw_answers, list):
                a_len_list.append(mean([len(a.split(" ")) for a in raw_answers]))
            else:
                a_len_list.append(len(raw_answers.split(" ")))

        if "evidence" in item:
            n_steps_list.append(len(item["evidence"]))
        elif "decomposition" in item:
            n_steps_list.append(len(item["decomposition"]))
        else:
            n_steps_list.append(0)

    q_len_mean = mean(q_len_list)
    a_len_mean = mean(a_len_list)
    n_steps_mean = mean(n_steps_list)

    print("Dataset : {}".format(dataset_path))
    print("Size : {}".format(len(raw_dataset)))
    print("Average length of questions : {}".format(q_len_mean))
    print("Average length of answers : {}".format(a_len_mean))
    print("Average number of steps : {}".format(n_steps_mean))

    return None


# Create dataset object before dataloader ...
class MyDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.data = []
        for item in dataset:
            if isinstance(item["answer"], list):
                answer = ", ".join(item["answer"])
            else:
                answer = str(item["answer"])

            new_item = {
                "question": item["question"],
                "answer": answer,
            }
            if "demos" in item:
                new_item["demos"] = item["demos"]
            if "contexts" in item:
                new_item["contexts"] = item["contexts"]

            self.data.append(new_item)

        self.length = len(dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, item: int):
        return self.data[item]


def setup_data_loader(data, seed: int = 42, max_num_worker: int = 2, batch_size: int = 1):
    # fix randomness of dataloader to ensure reproducibility
    # https://pytorch.org/docs/stable/notes/randomness.html
    fix_seed(seed)
    worker_seed = torch.initial_seed() % 2 ** 32
    print("worker_seed : {}".format(worker_seed))

    g = torch.Generator()
    g.manual_seed(worker_seed)

    dataloader_num_workers = multiprocessing.cpu_count()
    dataloader_num_workers = min(dataloader_num_workers, max_num_worker)
    print("dataloader_num_workers: " + str(dataloader_num_workers))

    dataset = MyDataset(data)

    dataloader = torch.utils.data.DataLoader(data_reader,
        dataset,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
        num_workers=dataloader_num_workers,
        generator=g,
        pin_memory=True)

    return dataloader


def answer_cleaning(dataset: str, pred: str, multiple_answer: bool = False):
    if dataset in ["feverous", "fool-me-twice", "hover", "strategy-qa"]:
        pred = pred.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,", " ", pred)
        pred = pred.split(" ")
        pred_answers = [i for i in pred if i in ("true", "false")]
    elif dataset in ["hotpot-qa", "cweb-qa"]:
        # remove the words in ()
        pred = pred.lower()
        pred = re.sub(r"\([^)]*\)", "", pred)
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,", " ", pred)
        yes_or_no_pred = pred.split(" ")
        # if yes_or_no_pred[0] in ["yes", "no"]:
        #     pred_answers = [yes_or_no_pred[0]]
        # else:

        yes_or_no_pred = [i for i in yes_or_no_pred if normalize_answer(i) in ("yes", "no")]

        # find all named entities in pred
        named_entities_spacy = nlp(pred).ents
        named_entities_nltk = process_an_item(pred)
        all_ne = []
        for ne in named_entities_spacy:
            all_ne.append(ne.text)
        for ne in named_entities_nltk:
            all_ne.append(ne)
        pred_answers = yes_or_no_pred + list(set(all_ne))
    else:
        raise ValueError("Invalid dataset name.")

    if multiple_answer:
        if len(pred_answers) == 0:
            pred_answers = [""]
        return pred_answers
    else:
        if len(pred_answers) == 0:
            pred_answers = ""
        else:
            pred_answers = pred_answers[0]
        return pred_answers


def extract_answer(raw_pred: str):
    if raw_pred == "":
        return []

    # find yes or no
    yes_no_answer = check_answer(raw_pred)
    if yes_no_answer is not None:
        return [yes_no_answer]

    # remove the words in ()
    raw_pred = re.sub("\([^)]*\)", "", raw_pred)
    # answers = re.split(",| and | or |/| either ", raw_pred)
    answers = re.split(", | and | or |/| either ", raw_pred)

    return answers













# def lemmatize_sentence(sentence):
#     lemmatizer = nltk.stem.WordNetLemmatizer()
#     lemmatized_sentence = []
#     for word, tag in nltk.pos_tag(nltk.word_tokenize(sentence)):
#         if tag.startswith('N'):
#             lemma = lemmatizer.lemmatize(word, pos='n')
#             lemmatized_sentence.append(lemma)
#     return " ".join(lemmatized_sentence)

def compute_f1(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_em(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def qa_evaluate(gold_answers: List[str], pred_answers: List[str]):
    em_scores = [0.0]
    f1_scores = [0.0]

    for pred_answer in pred_answers:
        for gold_answer in gold_answers:
            em_scores.append(compute_em(pred_answer, gold_answer))
            f1_scores.append(compute_f1(pred_answer, gold_answer))

    return max(em_scores), max(f1_scores)


def clean_passage(passage: str):
    raw_sentences = nltk.sent_tokenize(passage)
    # check if the last sentence is complete
    if len(raw_sentences) == 0:
        return ""

    # remove duplicate sentences and empty sentences
    sentences = []
    for sentence in raw_sentences:
        sentence = sentence.strip()
        if sentence in sentences:
            continue
        sentences.append(sentence)

    if sentences[-1].endswith("."):
        return " ".join(sentences)
    else:
        return " ".join(sentences[:-1])


def clean_entity(span):
    span = span.strip()
    if span == "":
        return None

    start, end = 0, len(span) - 1
    while start < len(span) and span[start] in expanded_punc_and_whitespace:
        start += 1

    while end >= 0 and span[end] in expanded_punc_and_whitespace:
        end -= 1

    return span[start:end + 1]


def build_decomposition_prompt(example):
    prompt = ""
    hops = example["hops"]
    if example["hop_type"] == "type_1":
        q1 = hops[0]["question"]
        a1 = re.escape(hops[0]["answer"]) + r'(?![a-zA-Z])'
        q2 = re.sub(a1, "#1", hops[1]["question"], flags=re.IGNORECASE)
        prompt += f"Q1: {q1}\nQ2: {q2}\n"
    elif example["hop_type"] == "type_2":
        q1 = hops[0]["question"]
        a1 = re.escape(hops[0]["answer"]) + r'(?![a-zA-Z])'
        q2 = re.sub(a1, "#1", hops[1]["question"], flags=re.IGNORECASE)
        a2 = re.escape(hops[1]["answer"]) + r'(?![a-zA-Z])'
        q3 = re.sub(a2, "#2", hops[2]["question"], flags=re.IGNORECASE)
        prompt += f"Q1: {q1}\nQ2: {q2}\nQ3: {q3}\n"
    elif example["hop_type"] == "type_3":
        q1 = hops[0]["question"]
        a1 = re.escape(hops[0]["answer"]) + r'(?![a-zA-Z])'
        q2 = hops[1]["question"]
        a2 = re.escape(hops[1]["answer"]) + r'(?![a-zA-Z])'
        q3 = re.sub(a1, "#1", hops[2]["question"], flags=re.IGNORECASE)
        q3 = re.sub(a2, "#2", q3, flags=re.IGNORECASE)
        prompt += f"Q1: {q1}\nQ2: {q2}\nQ3: {q3}\n"
    elif example["hop_type"] == "type_4":
        q1 = hops[0]["question"]
        a1 = re.escape(hops[0]["answer"]) + r'(?![a-zA-Z])'
        q2 = re.sub(a1, "#1", hops[1]["question"], flags=re.IGNORECASE)
        a2 = re.escape(hops[1]["answer"]) + r'(?![a-zA-Z])'
        q3 = re.sub(a2, "#2", hops[2]["question"], flags=re.IGNORECASE)
        a3 = re.escape(hops[2]["answer"]) + r'(?![a-zA-Z])'
        q4 = re.sub(a3, "#3", hops[3]["question"], flags=re.IGNORECASE)
        prompt += f"Q1: {q1}\nQ2: {q2}\nQ3: {q3}\nQ4: {q4}\n"
    elif example["hop_type"] == "type_5":
        q1 = hops[0]["question"]
        a1 = re.escape(hops[0]["answer"]) + r'(?![a-zA-Z])'
        q2 = hops[1]["question"]
        a2 = re.escape(hops[1]["answer"]) + r'(?![a-zA-Z])'
        q3 = re.sub(a1, "#1", hops[2]["question"], flags=re.IGNORECASE)
        q3 = re.sub(a2, "#2", q3, flags=re.IGNORECASE)
        a3 = re.escape(hops[2]["answer"]) + r'(?![a-zA-Z])'
        q4 = re.sub(a3, "#3", hops[3]["question"], flags=re.IGNORECASE)
        prompt += f"Q1: {q1}\nQ2: {q2}\nQ3: {q3}\nQ4: {q4}\n"
    elif example["hop_type"] == "type_6":
        q1 = hops[0]["question"]
        a1 = re.escape(hops[0]["answer"]) + r'(?![a-zA-Z])'
        q2 = re.sub(a1, "#1", hops[1]["question"], flags=re.IGNORECASE)
        a2 = re.escape(hops[1]["answer"]) + r'(?![a-zA-Z])'
        q3 = hops[2]["question"]
        a3 = re.escape(hops[2]["answer"]) + r'(?![a-zA-Z])'
        q4 = re.sub(a2, "#2", hops[3]["question"], flags=re.IGNORECASE)
        q4 = re.sub(a3, "#3", q4, flags=re.IGNORECASE)
        prompt += f"Q1: {q1}\nQ2: {q2}\nQ3: {q3}\nQ4: {q4}\n"
    else:
        raise ValueError("Invalid hop type")
    return prompt











































