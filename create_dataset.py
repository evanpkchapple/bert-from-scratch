import re
import os
import pandas as pd
import nltk
import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen

nltk.download('punkt')

def clean_text(text):
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r'\\[nrt]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_into_sentences(text):
    return nltk.sent_tokenize(text)

output_dir = 'gutenberg_texts'
os.makedirs(output_dir, exist_ok=True)

df_metadata = pd.read_csv('gutenberg_metadata.csv') # Download this from the provided link
# https://www.kaggle.com/datasets/mateibejan/15000-gutenberg-books

for key, row in df_metadata.iterrows():
    author = row['Author']
    title = row['Title']
    link = row['Link']
    bookshelf = row['Bookshelf']
    book_id = int(link.split('/')[-1])

    try:
        safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
        filename = f"{book_id}_{safe_title[:50]}.txt"
        file_path = os.path.join(output_dir, filename)

        if os.path.exists(file_path):
            print(f"Already exists, skipping: {filename}")
            continue

        page = requests.get(link)
        soup = BeautifulSoup(page.content, 'html.parser')

        text_link = 'http://www.gutenberg.org' + soup.find("a", string="Plain Text UTF-8")['href']

        http_response_object = urlopen(text_link)
        raw_text = http_response_object.read().decode('utf-8', errors='ignore')
        raw_text = clean_text(raw_text)

        sentences = split_into_sentences(raw_text)

        with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(sentence) > 1:
                    f.write(sentence + "\n")

        print(f"Saved: {filename}")
    except Exception as e:
        print(f"Failed to acquire or save {title} (ID {book_id}): {e}")
