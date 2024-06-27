import re
import sys
from random import shuffle
from nltk.tokenize import sent_tokenize

def read_dump(fpath):
    print(f">> DATA: WIKI READING DUMP: {fpath}")
    typ = 'THK'
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
                    #p = f"<a type={typ} idx={idx}-{c}>\n<par>{l}</par>\n</a>\n"
                    p = f"<a type={typ}>\n{l}\n</a>\n"
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

    with open('sample_wiki_reading.txt','w',encoding='utf-8') as fout:
        for i in ids:
            fout.write(paragraphs[i])


def get_user_categories(fpath):
    categories = {}
    with open(fpath, encoding="utf-8") as fin:
         for l in fin:
            l = l.rstrip('\n')
            category, questions = l.split(':')
            questions = re.split(';\s*',questions)
            categories[category] = questions
    return categories


def mk_questions(categories):
    qas = []
    for category, questions in categories.items():
        print(f">> DATA: WIKI MAKING QUESTIONS FOR CATEGORY: {category}")
        #print(f">> DATA: WIKI QUESTIONS: {questions}")
        typ = 'TLK'
        category = category.replace(' ','_')
        fpath = f"data/simple/categories/{category}/linear.{category.lower()}.doc.txt"
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
                        for question in questions:
                            q = question.replace('[]',title)
                            answer = sent_tokenize(l)[0]
                            p = f"<a type={typ}>\n<u speaker=HUM>{q}</u>\n<u speaker=BOT>{answer}</u>\n</a>\n"
                            #print(p)
                            qas.append(p)
        print(f">> DATA: WIKI FOUND {len(qas)} QUESTION-ANSWER PAIRS.")
    return qas


def sample_from_categories_for_defining():
    nsamples = int(sys.argv[1])
    cat_questions_path = sys.argv[2]
    categories = get_user_categories(cat_questions_path)
    #categories = {"Vegetables": "What kind of vegetable is [MASK]?"}
    questions = mk_questions(categories)
    ids = list(range(len(questions)))
    shuffle(ids)
    ids = ids[:nsamples]

    with open('sample_wiki_defining.txt','w',encoding='utf-8') as fout:
        for i in ids:
            fout.write(questions[i])


#sample_from_dump_for_reading()
sample_from_categories_for_defining()
