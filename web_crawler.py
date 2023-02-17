import requests
import math
from bs4 import BeautifulSoup
from queue import Queue
from pprint import pprint
import urllib.request
import random
import re
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import pickle
import json

starter_url = "https://gbdev.io/resources.html#introduction"

q = Queue(maxsize = 0)
visited = []
custom_stop_words = ["community", "pan", "docs"]
top_ten = []
clean_corpus = []


# fake user agents to assist the web crawler with being able to access sites
user_agents_list = [
    'Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.83 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36'
]

# function that gets the pagedata and returns it as a soup object
def gather_soup_ingredients(page):
    try:

        r = requests.get(page)
        data = r.text
        soup = BeautifulSoup(data, 'html.parser')
    except urllib.error.HTTPError:
        print("ERROR 404")
    return soup

# function that cleans up a url to a nice readable format
def clean_up_url(link_str):
    if link_str.startswith('/url?q='):
        link_str = link_str[7:]
        print('MOD:', link_str)
    if '&' in link_str:
        i = link_str.find('&')
        link_str = link_str[:i]
    return link_str


# criteria to determine if a link is worth adding to the crawling queue
def link_Is_Relevant(link_str):
    if 'GameBoy' in link_str or 'gameboy' in link_str or 'gameBoy' in link_str or 'Gameboy' in link_str:
        if link_str.startswith('http') and \
            'google' not in link_str and \
            'twitter' not in link_str and \
            'facebook' not in link_str and \
            'github' not in link_str and \
            '.html' not in link_str and \
            '.txt' not in link_str and \
            '.de/' not in link_str and \
            'imrannazar' not in link_str and \
            'log-in?redirect' not in link_str and \
            '.pdf' not in link_str and \
            '.jpg'  not in link_str:
            return True
        else:
            return False
    else: 
        return False

# function to crawl a single web page. 
def crawl_single_page(page, f):
    visited.append(page)
    soup = gather_soup_ingredients(page)
    f.write("\n\n\nHERE ARE THE LINKS FROM PAGE: "+ page + "\n")

    # for all the links in the soup object, clean them up and add them to the queue
    # if they are relevant
    for link in soup.find_all('a'):
        raw_link = str(link.get('href'))
        link_str = clean_up_url(raw_link)
        if link_Is_Relevant(link_str):
            if link_str not in visited: #check to see if the link has already been crawled
                q.put(link_str)         # if it hasn't, add it to the queue and write to urls.txt
                f.write(link_str + '\n')


# general crawling function. Will crawl until the queue is empty, or 
# desired number of relevant links have been found
def crawl(link_threshold):
    with open('urls.txt', 'w') as f:
        while (len(visited) < link_threshold) and not q.empty():
            link = q.get()
            crawl_single_page(link, f)

# helper function to print the queue, during debugging      
def print_queue():
    print(list(q.queue))

# helper function to print the visited pages, during debugging
def print_visited():
    print("\n\nVISITED LINKS WERE AS FOLLOWS: \n")
    pprint(visited)

# helper function to read all the urls saved in
def read_urls():
    with open('urls.txt', 'r') as f:
        urls = f.read().splitlines()
        for u in urls:
            print(u)
                
# function to determine if an element is visible
def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match('<!--.*-->', str(element.encode('utf-8'))):
        return False
    return True

#2. Write a function to loop through your URLs and scrape all text off each page. Store each
#page’s text in its own file.
def download_raw_corpus(): 

    for idx, url in enumerate(visited): 
        try:
            filename = str(idx) + ".txt"
            with open( filename, 'w') as f:
                html = requests.get(url, headers={'User-Agent': random.choice(user_agents_list)})
                soup = BeautifulSoup(html.text, 'html.parser')
                data = soup.findAll(text=True)
                result = filter(visible, data)
                temp_list = list(result)      # list from filter
                temp_str = ' '.join(temp_list)
                print(temp_str)
                f.write(temp_str)
        except urllib.error.HTTPError: 
            print("Unable to access " + url + "\nAccess is forbidden.\n")
        except TypeError as e: 
            print(e)



#3. Write a function to clean up the text from each file. You might need to delete newlines
#and tabs first. 
def clean_file(file):
    cleaned = file.replace('\n', '')
    cleaned = cleaned.replace('\t', '')
    cleaned = cleaned.replace('\r','')
    cleaned = " ".join(word_tokenize(cleaned))
    return cleaned

# Loop through documents in corpus and clean up the files
def clean_up_corpus():
    cleaned_file = ''
    sentences = ''

    # Extract sentences with NLTK’s sentence tokenizer. Write the sentences for
    # each file to a new file. That is, if you have 15 files in, you have 15 files out.
    for idx, url in enumerate(visited): 
        unclean_filename = str(idx) + ".txt"
        clean_filename = str(idx) + "_clean.txt"
        with open(unclean_filename, 'r') as f: 
            data = f.read()
            cleaned_file = clean_file(data)
            try:
                sentences = sent_tokenize(cleaned_file)
            except TypeError as e:
             print(e)

        with open(clean_filename, 'w') as f:
            for sentence in sentences:
                f.write(sentence + '\n')
        
# The following 4 functions are helper functions to pre-process the documents in the corpus
def remove_aposterphe(data):
    return np.char.replace(data, "'", "").tolist()


def remove_punctuation(data):
    symbols = "\"#$%&()*+-/:;<=>@[\]^_`{|}~\n•?.!"
    for i in symbols:
        data = np.char.replace(data, i, '')
    return_string = np.array2string(data)
    return return_string

def remove_sentence_end(data):
    symbols = "!.?"
    for i in symbols:
        data = np.char.replace(data, i, '')
    return_string = np.array2string(data)
    return return_string

def remove_stop_words(data):
    filtered = ""
    words = word_tokenize(data)
    for word in words: 
        if word not in stopwords.words('english'):
            filtered = filtered + " " + word
    return filtered

def remove_single_characters(data):
    words = word_tokenize(data)
    new_text = ""
    for w in words:
        if len(w) > 1:
            new_text = new_text + " " + w
    return new_text

def remove_extra_text(data):
    sent = sent_tokenize(data)
    new_text = []
    for s in sent:
        if "open in new window" in s:
            new_text.append(s.replace('open in new window',''))
        else: 
            new_text.append(s)
    return new_text        

# Using the pre-process helper functions, pre-process the documents in the corpus
def preprocess_data(file):
    data = file.lower()
    data = remove_punctuation(data)
    data = remove_aposterphe(data)
    data = remove_single_characters(data)
    data = remove_sentence_end(data)
    data = remove_stop_words(data)
    return data

# use TF-IDF measure to determine important words in the corpus.
# This Routine does the following: 
# 1. reads in the cleaned text files
# 2. runs the pre-processing on the files
# 3. computes the TF-IDF of the tokesn in the corpus
# 4. Determines the top 40 terms in the corpus
# 5. Writes the top 40 terms to a file
def extract_important_terms():
    documents = []
    corpus = []

    # get all the sentences from all the files
    for idx in range(15):
        filename = str(idx) + "_clean.txt"
        with open (filename, 'r') as f:
            data = f.read()
        documents.append(data)

    # preprocess the data
    for idx in range(15):
        corpus.append(preprocess_data(documents[idx]))
    
    # calculate the TF-IDF
    tf_idf_dict = calculate_tf_idf(corpus)

    # order the results by tf-idf score
    # print the top 40 results
    top_forty = sorted(tf_idf_dict, key=tf_idf_dict.get, reverse=True)[:40]
    print("Top 40 terms are: ", top_forty)

    # write top 40 results to a file
    results = open("top40.txt", 'w')
    with open("top40.txt", 'w') as f:
        for result in top_forty:
            f.write(result + '\n')

# Run the actual calculation of the following: 
# 1. Term Frequency
# 2. Inverse Document Frequency
# 3. TF-IDF of the terms in the corpus
def calculate_tf_idf(corpus):
    word_dicts = create_word_dicts(corpus)
    tf_per_document = []
    idfs = {}
    tf_idf_dict = {}

    for i in range(len(corpus)):
        tf_per_document.append(tf(word_dicts,corpus, i))

    idfs = idf(word_dicts)

    tf_idf_dict = tfidf(tf_per_document,idfs)
    return tf_idf_dict

# Given TF and IDF, calculate the TF-IDF
def tfidf(tfs, idfs):
    tfidf = {}
    for i in range(len(tfs)):
        for word, val in tfs[i].items():
            tfidf[word] = val*idfs[word]
    return(tfidf)

# Given a dictionary of words in the corpus, the corpus, and a document index, calculate TF
def tf(word_dict, corpus, index):
    tfDict = {}

    for word, count in word_dict[index].items():
        tfDict[word] = count/float(len(corpus))
    return(tfDict)

# Given a dictionary of words in the corpus, determine the IDF of each word.
def idf(word_dict):
    idfDict = {}
    N = len(word_dict)
    
    idfDict = dict.fromkeys(word_dict[0].keys(), 0)
    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / (float(val) + 1))
    return(idfDict)


# Given the corpus, create a dictionary of unique words per document
def create_word_dicts(corpus):
    total_words = []
    word_dicts = []
    # create one long string from the corpus of unique invidivual words:
    for document in corpus: 
        words = document.split()
        for word in words:
            if word not in total_words:
                total_words.append(word)
    
    # creates a dictionary of unique words in the corpus and their occurences in each document
    for idx, document in enumerate(corpus):
        word_dicts.append(dict.fromkeys(total_words, 0))
        doc_words = document.split()
        for word in doc_words:
            word_dicts[idx][word]+=1
    return word_dicts


#5. Manually determine the top 10 terms from step 4, based on your domain knowledge.
def define_top_ten():
    return ["gameboy", "game", "score", "gbdk", "development", "nes", "software", "programming", "assembly", "tutorials"]


# knowledge base is in the form of {(key, value)} as 
# {(term, some sentences about a term from the corpus)}
def build_knowledge_base(terms):
    print("Top Ten Terms: ", terms)
    knowledge_base = dict.fromkeys(terms, [])
    sent = []
    term_in_sent = []
    corpus = [] # each element is a whole document as a string
    clean_corpus = []

    for idx in range(15):
        filename = str(idx) + "_clean.txt"
        with open (filename, 'r') as f:
            data = f.read()
        corpus.append(data)
    
    # write base corpus, no pre-processing to file for debugging
    with open('basecorpus.txt', 'w') as f: 
        for document in corpus:
            f.write(document)

    # clean the corpus
    for idx in range(len(corpus)):
        data = corpus[idx] # data is a whole document as a string
        data = data.lower() # data is a lowercase whole document as a string
        data = remove_extra_text(data) # data is now a list of sentences
        for datum in data: 
            clean_corpus.append(datum)

    # write clean corpus to file for debugging
    with open('cleancorpus.txt', 'w') as f: 
        for document in clean_corpus:
            f.write(document+"\n")

    # tokenize the sentences in the cleaned corpus
    for document in clean_corpus: 
        tokens = sent_tokenize(document)
        for token in tokens:
            sent.append(token)

    # if the term is in a sentence in the corpus, add the sentence to the knowledge base
    for term in terms:
        term_in_sent = []
        for sentence in sent:
            if term in sentence:
                term_in_sent.append(sentence)
        knowledge_base[term] = term_in_sent


    print("len of assembly sentences is : ", len(knowledge_base["assembly"]))
    print(knowledge_base['assembly'][8])

    # prints the knowledge base in text format
    with open('kb.txt', 'wt') as f:
        pprint(knowledge_base, stream=f)

    # Pickle Knowledge Base
    with open('knowledge_base.pickle', 'wb') as handle:
        pickle.dump(knowledge_base, handle, protocol=pickle.HIGHEST_PROTOCOL)

        

# DO NOT DELETE - GETS URLS
#q.put(starter_url)
#crawl(15)
#print_visited()
#download_raw_corpus()
clean_up_corpus()
extract_important_terms()
top_ten = define_top_ten()
build_knowledge_base(top_ten)