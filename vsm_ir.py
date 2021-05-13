import math
import os
import string
import sys
from lxml import etree
import nltk
import json
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

# word : {file_number : how many times word in file_number}
inverted_index = {}
nltk.download('stopwords')
nltk.download('punkt')
FREQ_KEY = 'FREQ'
TF_IDF_KEY = 'TF-IDF-SCORE'


def count_word_in_text(record_number, record_text):
    punc_list = '-.;:!?/\,#@$&)(\'"'
    replacement = ' ' * len(punc_list)
    remove_digits = str.maketrans('', '', string.digits)
    record_text = record_text.translate(remove_digits)
    record_text = record_text.translate(str.maketrans(punc_list, replacement))
    text_tokens = word_tokenize(record_text)
    tokens_without_sw = [word.lower() for word in text_tokens if not word.lower() in stopwords.words('english')]
    for token in tokens_without_sw:
        if token in inverted_index:
            record_dict = inverted_index[token]
            if record_number in record_dict:
                record_dict[record_number][FREQ_KEY] += 1
            else:
                record_dict[record_number] = {FREQ_KEY: 1}
        else:
            inverted_index[token] = {record_number: {FREQ_KEY: 1}}


def parse_one_xml_file(doc):
    for record in doc.xpath(".//RECORD"):
        record_number = " ".join(record.xpath(".//RECORDNUM/text()"))
        record_text = ""
        record_text += " " + " ".join(record.xpath(".//TITLE/text()"))
        record_text += " " + " ".join(record.xpath(".//EXTRACT/text()"))
        record_text += " " + " ".join(record.xpath(".//ABSTRACT/text()"))
        # record_text += " " + " ".join(record.xpath(".//TOPIC/text()"))
        count_word_in_text(record_number, record_text)


def update_tfidf_scores():
    doc_to_max_freq = {}
    for word in inverted_index:
        for doc_num in inverted_index[word]:
            if doc_num in doc_to_max_freq:
                if inverted_index[word][doc_num][FREQ_KEY] > doc_to_max_freq[doc_num]:
                    doc_to_max_freq[doc_num] = inverted_index[word][doc_num][FREQ_KEY]
            else:
                doc_to_max_freq[doc_num] = inverted_index[word][doc_num][FREQ_KEY]

    num_documents = len(doc_to_max_freq)

    for word in inverted_index:
        for doc_num in inverted_index[word]:
            tf = inverted_index[word][doc_num][FREQ_KEY] / doc_to_max_freq[doc_num]
            idf = math.log2(num_documents / len(inverted_index[word]))
            inverted_index[word][doc_num][TF_IDF_KEY] = tf * idf


def build_inverted_index(path):
    for filename in os.listdir(path):
        root = etree.parse(path + "\\" + filename)
        parse_one_xml_file(root)

    update_tfidf_scores()
    # Serializing json
    json_inverted_index = json.dumps(inverted_index)

    # Writing to sample.json
    with open("vsm_inverted_index.json", "w") as outfile:
        outfile.write(json_inverted_index)
    outfile.close()


def parse_inverted_index(path):
    with open(path) as f:
        data = json.load(f)

    inverted_index.update(data)


def build_query_vector(query):
    query_vector = {}
    punc_list = '-.;:!?/\,#@$&)(\'"'
    replacement = ' ' * len(punc_list)
    remove_digits = str.maketrans('', '', string.digits)
    query_text = query.translate(remove_digits)
    query_text = query_text.translate(str.maketrans(punc_list, replacement))
    text_tokens = word_tokenize(query_text)
    tokens_without_sw = [word.lower() for word in text_tokens if not word.lower() in stopwords.words('english')]
    for token in tokens_without_sw:
        if token in query_vector:
            query_vector[token] += 1
        else:
            query_vector[token] = 1
    return query_vector


def print_relevant_documents(query):
    query_vector = build_query_vector(query)
    query_vector_weight = 0
    doc_to_score = {}
    for word in query_vector:
        query_vector_weight += query_vector[word] ** 2
        if word in inverted_index:
            for doc_num in inverted_index[word]:
                if doc_num in doc_to_score:
                    doc_to_score[doc_num] += inverted_index[word][doc_num][TF_IDF_KEY] * query_vector[word]
                else:
                    doc_to_score[doc_num] = inverted_index[word][doc_num][TF_IDF_KEY] * query_vector[word]

    doc_to_weight = {}
    for word in inverted_index:
        for doc_num in inverted_index[word]:
            if doc_num in doc_to_weight:
                doc_to_weight[doc_num] += inverted_index[word][doc_num][TF_IDF_KEY] ** 2
            else:
                doc_to_weight[doc_num] = inverted_index[word][doc_num][TF_IDF_KEY] ** 2

    for doc_num in doc_to_score:
        doc_to_score[doc_num] = doc_to_score[doc_num] / (math.sqrt(query_vector_weight * doc_to_weight[doc_num]))

    sorted_docs = sorted(doc_to_score, key=lambda k: doc_to_score[k], reverse=True)
    write_file = open('ranked_query_docs.txt', 'w')
    for doc_num in sorted_docs:
        if doc_to_score[doc_num] > 0:
            write_file.write(doc_num + '\n')
    write_file.close()


def main():
    if sys.argv[1] == "create_index":
        path = sys.argv[2]
        build_inverted_index(path)
    if sys.argv[1] == "query":
        inverted_index_path = sys.argv[2]
        parse_inverted_index(inverted_index_path)
        query = sys.argv[3]
        print_relevant_documents(query)


if __name__ == '__main__':
    main()
