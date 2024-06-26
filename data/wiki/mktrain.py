import re
import sys
from random import shuffle
from nltk.tokenize import sent_tokenize

def read_dump(fpath):
    print(f">> DATA: WIKI READING DUMP: {fpath}")
    paragraphs = []
    idx=0
    c = 0
    with open(fpath, encoding='utf-8') as fin:
        for l in fin:
            l = l.rstrip('\n')
            if l.startswith('<doc'):
                m = re.search('doc id="([^ ]*)"',l)
                idx = m.group(1)
            elif l.startswith('</doc'):
                c = 0
            else:
                c+=1
                if l != "" and not l.startswith('##') and len(l.split()) > 5:
                    p = f"<a type=REA idx={idx}-{c}>\n<par>{l}</par>\n</a>\n"
                    paragraphs.append(p)
    print(f">> DATA: WIKI FOUND {len(paragraphs)} PARAGRAPHS")
    return paragraphs


def sample_from_dump_for_reading():
    nsamples = int(sys.argv[1])

    wiki_path = 'data/simple/simplewiki-latest-pages-articles.xml.raw.doc.txt'
    paragraphs = read_dump(wiki_path)
    ids = list(range(len(paragraphs)))
    shuffle(ids)
    ids = ids[:nsamples]

    with open('train.txt','w',encoding='utf-8') as fout:
        for i in ids:
            fout.write(paragraphs[i])


def mk_questions(category, question):
    print(f">> DATA: WIKI MAKING QUESTIONS FOR CATEGORY: {category}")
    fpath = f"data/simple/categories/{category}/linear.{category.lower()}.doc.txt"
    questions = []
    firstline = ""
    idx=0
    with open(fpath, encoding='utf-8') as fin:
        for l in fin:
            l = l.rstrip('\n')
            if l.startswith('<doc'):
                m = re.search('id="([^ ]*)"',l)
                idx = m.group(1)
                m = re.search('title="([^"]*)"',l)
                title = m.group(1)
            elif l.startswith('</doc'):
                firstline = ""
            else:
                if len(firstline) > 0:
                    continue
                if l != "" and not l.startswith('##') and len(l.split()) > 5:
                    firstline = l
                    q = question.replace('[MASK]',title)
                    answer = sent_tokenize(l)[0]
                    p = f"<a type=SKT idx={idx}>\n<u speaker=HUM>{q}</u>\n<u speaker=BOT>{answer}</u>\n</a>\n"
                    print(p)
                    questions.append(p)
    print(f">> DATA: WIKI FOUND {len(questions)} QUESTION-ANSWER PAIRS.")

mk_questions("Vegetables", "What kind of vegetable is [MASK]?")
