from posixpath import basename
import sys
import os
from rouge import Rouge
import numpy as np
import spacy
from spacy.tokenizer import Tokenizer
from spacy import displacy
from spacy import tokens
from spacy.tokens import DocBin
nlp = spacy.load("en_core_web_sm")
from spacy.lang.en.stop_words import STOP_WORDS
import string
import pandas as pd
scores = {}
gold_path = "path_to_folder_of_documents"
auto_path = "path_to_folder_of_all_auto-summary"
text_files = [f for f in os.listdir(auto_path) if f.endswith('.txt')]
folders = [d for d in os.listdir(gold_path) if os.path.isdir(os.path.join(gold_path, d))]
gold_file="gold.txt"
for text_file in text_files:
    # Get the base name of the text file (without extension)
    base_name = os.path.splitext(text_file)[0]
    # Check if there is a folder with the same name as the text file
    if base_name in folders:
        # print(base_name)
        # Define the path to the matching folder
        folder_path = os.path.join(gold_path, base_name)
        # print(folder_path)
        for item in folders:
            if base_name==item:
                # print(base_name)
                with open(f"{auto_path}/{text_file}", 'r') as f1:
                    a = f1.read().translate(str.maketrans('', '', string.punctuation))
                model = a.lower()
                # print ("print mode",model)
                with open(f"{folder_path}/{gold_file}", 'r') as f2:
                    gold_read = f2.read().translate(str.maketrans('', '', string.punctuation))
                gold = gold_read.lower()
                # print("print gold",gold)
                rouge = Rouge()
                #make sure name of auto summary and folder contain gold summary are the same
                name=text_file+"_and_"+item
                scores[name]=rouge.get_scores(model, gold)

# print(scores)
R1_f_score=[]
R2_f_score=[]
RL_f_score=[]
for ke, val in scores.items():
    for it in val:
        for k1, v1 in it.items():
            if k1==list(it.keys())[0]:
                R1_f=list(v1.values())[2]
                R1_f_score.append(R1_f)
            if k1==list(it.keys())[1]:
                R2_f=list(v1.values())[2]
                R2_f_score.append(R2_f)
            if k1==list(it.keys())[2]:
                RL_f=list(v1.values())[2]
                RL_f_score.append(RL_f)

# print(R1_f_score)
# print(R2_f_score)
# print(RL_f_score)
# I only save F-score result
df = pd.DataFrame({
    'R1_f': R1_f_score,
    'R2_f': R2_f_score,
    'RL_f': RL_f_score
})
#the result is for only one gold summary. DUC2004 contains 5 gold summary of each cluster
df.to_csv('gold.csv', index=False)