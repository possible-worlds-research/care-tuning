""" Fine-tune GPT2 on CPU

Usage:
  fine_tuning_gpt.py tune --model=<path> --data=<path> --size=<numsents> [--start=<n>]
  fine_tuning_gpt.py test --model=<path>
  fine_tuning_gpt.py livetune --model=<path>
  fine_tuning_gpt.py (-h | --help)
  fine_tuning_got.py --version

Options:
  -h --help         Show this screen.
  --version         Show version.
  tune              Launch fine-tuning process on corpus.
  test              Converse with human
  livetune          Fine-tune model during conversation.
  --data=<path>     Path of data for fine-tuning.
  --size=<numsents> How much of the data to use in fine-tuning (in sentences).
  --start=<n>       Optional: where to start in the corpus (in number of sentences).
  --model=<path>    Path of model *directory* (to further tune or to test on).

"""


import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import re
import string
import sys
import random
import math
from os.path import join
from pathlib import Path
from datetime import datetime
from docopt import docopt
from time import sleep
import collections
import numpy as np
import tensorflow as tf
from transformers import logging
logging.set_verbosity_error()


from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import create_optimizer
from transformers import pipeline
from transformers import TextDataset,DataCollatorForLanguageModeling
from datasets import Dataset, load_dataset

family = {"BOT":"<bot'(x)>", "HUM":"<human'(x)>"}

def read_corpus(filename, size=1000, start=None):
    corpus = []
    c = 0
    with open(filename) as f:
        for l in f:
            l = l.rstrip('\n')
            if '<doc' not in l and '</doc' not in l:
                if start != None:
                    if c < start:
                        c+=1
                    else:
                        c = 0
                        start = None
                else:
                    if len(l) > 0 and '##' not in l:
                        corpus.append(l)
                        c+=1
            if start is None and c == size:
                break
    return corpus

def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

def mk_train_test(lm_dataset):

    if len(lm_dataset) > 10:
        train_size = int(0.7*len(lm_dataset))
        test_size = int(0.2*len(lm_dataset))

        downsampled_dataset = lm_dataset.train_test_split(train_size=train_size, test_size=test_size, seed=42)
        #print(downsampled_dataset)

        tf_train_dataset = downsampled_dataset["train"]
        tf_eval_dataset = downsampled_dataset["test"]
    else:
        tf_train_dataset = lm_dataset
        tf_eval_dataset = lm_dataset

    return tf_train_dataset, tf_eval_dataset


def process_text(dataset):
    def gen():
        for i in range(len(dataset)):
            text = dataset[i]
            rest = len(text.split()) % chunk_size
            if rest != 0:
                yield {"text": ' '.join(text.split()[:-rest])}
                yield {"text": ' '.join(text.split()[rest:])}
            else:
                yield {"text": text}
                yield {"text": text}

    sample = Dataset.from_generator(gen)
    tokenized_dataset = sample.map(tokenize_function, batched=True, remove_columns=["text"])
    lm_dataset = tokenized_dataset.map(group_texts, batched=True)
    return lm_dataset


def fine_tune(model, lm_dataset, lr=0.00001):
    tf_train_dataset, tf_eval_dataset = mk_train_test(lm_dataset)

    training_args = TrainingArguments(
    output_dir=args['--model'],
    evaluation_strategy="epoch",
    learning_rate=lr,
    weight_decay=0.001,
    push_to_hub=False,
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tf_train_dataset,
    eval_dataset=tf_eval_dataset,
    data_collator=data_collator,
    )

    trainer.train()
    return trainer.model


def predict(prompt, m, t):
    pp = pipeline('text-generation', model=m, tokenizer=t, config={'max_length':20})
    answer = pp(prompt)[0]['generated_text'].replace('\n',' ')[len(prompt):] #don't repeat prompt
    #print("PROMPT",prompt)
    #print("ANSWER",answer)
    answer = answer.split('</u>')[0]
    return answer


def resolve_indexicals(conversation):
    #print("RESOLVING INDEXICALS IN",conversation)
    resolved = ""
    speakers = re.findall('<u speaker=(...)>', conversation)
    turns = re.split('</u>', conversation)
    for turn in turns:
        try:
            speaker = re.findall('<u speaker=(...)>', turn)[0]
        except:
            continue
        interlocutor = [s for s in speakers if s != speaker][0]
        utterance = re.sub('<u speaker=(...)>', '', turn)
        if utterance.isspace() or utterance == '':
            continue
        if utterance.startswith('I '):
            utterance = family[speaker] + utterance[2:]
        if ' I ' in utterance:
            utterance = utterance.replace(' I ', f' {family[speaker]} ')
        for p in ['.',',',';',':','?','!']:
            if ' me'+p in utterance:
                utterance = utterance.replace(' me'+p,  f' {family[speaker]}'+p)
        if utterance.startswith('You '):
            utterance = interlocutor + utterance[4:]
        if ' you ' in utterance:
            utterance = utterance.replace(' you ', f' {family[interlocutor]} ')
        for p in ['.',',',';',':','?','!']:
            if ' you'+p in utterance:
                utterance = utterance.replace(' you'+p,  f' {family[interlocutor]}'+p)
        resolved+='<u speaker='+speaker+'>'+utterance+'</u>'
    #print("RESOLVED:", resolved)
    return resolved



def fine_tune_during_conversation(model):
    date_string = f'{datetime.now():%Y-%m-%d %H:%M}'
    conversation = open(join('data','conversations', date_string+'.txt'),'w')
    training_data = ""

    human = input("Please enter your 3-letter speaker code: ").rstrip('\n')

    turns = []
    answer = '<u speaker=BOT>Hello!</u>'
    turns.append(answer)
    prompt = input(human+">> ").rstrip('\n')
    if prompt != 'q':
        prompt = '<u speaker='+human+'>'+prompt+'</u><u speaker=BOT>'

    while prompt != 'q':
        prompt_no_markers = prompt.replace('XXX ','') #Turn 0
        answer = predict(prompt_no_markers, model, tokenizer) #Turn 1
        print("BOT>> "+answer)
        turns.append(prompt_no_markers.replace('<u speaker=BOT>',''))
        turns.append('<u speaker=BOT>'+answer+'</u>')
        if 'XXX ' in prompt:
            #anticipating what the human said
            turns[0] = prompt_no_markers.replace('<u speaker=BOT>','').replace('='+human,'=BOT')
        training_data+=''.join(turns[:2])
        turns.clear()
        turns.append('<u speaker=BOT>'+answer+'</u>')
        prompt = input(human+">> ").rstrip('\n')
        if prompt != 'q':
            prompt = '<u speaker='+human+'>'+prompt+'</u><u speaker=BOT>'
        if len(training_data) > 200:
            training_data = resolve_indexicals(training_data)
            training_data = training_data.replace('</u><u speaker='+human, '</u>\n<u speaker='+human)
            conversation.write(training_data)
            #lm_dataset = process_text([training_data])
            #model = fine_tune(model, lm_dataset, lr=0.000001)
            training_data = ""
    training_data = resolve_indexicals(training_data)
    training_data = training_data.replace('</u><u speaker='+human, '</u>\n<u speaker='+human)
    conversation.write(training_data)
    conversation.close()


def fine_tune_from_corpus(model, corpus_path, size=1000, start=None):

    def ft_iter(model, corpus, iter_size):
        def gen():
            for i in range(iter_size):
                yield {"text": corpus[i]}

        sample = Dataset.from_generator(gen)
        tokenized_dataset = sample.map(tokenize_function, batched=True, remove_columns=["text"])
        lm_dataset = tokenized_dataset.map(group_texts, batched=True)

        #print("\n>> Now fine-tuning...\n")
        ft_model = fine_tune(model, lm_dataset)
        #print("\n>> Finished fine-tuning.\n")
        return ft_model
    
    corpus = read_corpus(corpus_path, size=size, start=start)
    iter_size = 2 # Number of sentences considered at any one time

    for i in range(0,len(corpus),iter_size):
        try:
            print("\n>>>",i,'\n'.join(corpus[i:i+iter_size]))
            model = ft_iter(model, corpus[i:i+iter_size], iter_size)
            if corpus[i].startswith('<u speaker'):
                prompt = corpus[i].split('</u>')[0]+'</u><u speaker=BOT>'
                answer = predict(prompt, model, tokenizer)
            else:
                answer = predict(' '.join(corpus[i].split()[:5]), model, tokenizer)
            print(answer)
            sleep(10)
        except:
            print("ERROR >>>",corpus[i])
    model.save_pretrained(args['--model'])


def chat(ft_model):
    text = input(">> ").rstrip('\n')
    text = '<u speaker=HUM>'+text+'</u><u speaker=BOT>'

    while text != 'q':
        answer = predict(text, ft_model, tokenizer)
        print(answer)
        text = input(">> ").rstrip('\n')
        if text != 'q':
            text = '<u speaker=HUM>'+text+'</u><u speaker=BOT>'


if __name__ == "__main__":
    args = docopt(__doc__, version='GPT2 Fine-tuning v0.1')
    print(args)
    
    Path("data/conversations/").mkdir(parents=True, exist_ok=True)
    orig_model_checkpoint = "gpt2"

    if os.path.isdir(args['--model']):
        model_checkpoint = args['--model']
        print("Loading model from ", model_checkpoint)
    else:
        model_checkpoint = "gpt2"
        print("Loading standard gpt2 model.")
    chunk_size = 30
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(orig_model_checkpoint)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="pt")
    tokenizer.pad_token = tokenizer.eos_token

    if args['tune']:
        data = args['--data']
        size = int(args['--size'])
        if args['--start']:
            start = int(args['--start'])
        else:
            start = None
        ft_model = fine_tune_from_corpus(model, data, size=size, start=start)

    if args['test']:
        chat(model)


    if args['livetune']:
        fine_tune_during_conversation(model)

