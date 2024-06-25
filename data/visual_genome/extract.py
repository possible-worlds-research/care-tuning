import io
import json
import requests
import zipfile
from tqdm import tqdm
from os.path import join

host_url = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/"

def download_qa(fpath):
    response = requests.get(join(host_url, fpath), stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(fpath, mode="wb") as f:
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))
                f.write(chunk)
    print(fpath,"successfully downloaded")

fpath = 'question_answers.json.zip'
download_qa(fpath)

with zipfile.ZipFile(fpath) as zf:
    with io.TextIOWrapper(zf.open(fpath.replace('.zip','')), encoding="utf-8") as f:
    #with open(fpath, "r", encoding="utf-8") as fi:
     content = json.load(f)


for item in content:
    qas = item['qas']
    for qa in qas:
        question = qa['question']
        answer = qa['answer']
        if len(answer.split()) > 3:
            print(f"<u speaker=HUM>{question}</u>\n<u speaker=BOT>{answer}</u>\n")
