import math
import os
import string
import sys
from lxml import etree
import nltk
import json
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from nltk.tokenize import word_tokenize

# word : {file_number : how many times word in file_number}
inverted_index = {}
nltk.download('stopwords')
nltk.download('punkt')
FREQ_KEY = 'FREQ'
TF_IDF_KEY = 'TF-IDF-SCORE'

porter_stemmer = PorterStemmer()


def count_word_in_text(record_number, record_text):
    record_number = record_number.lstrip("0").rstrip()
    punc_list = '-.;:!?/\,#@$&)(\'"'
    replacement = ' ' * len(punc_list)
    remove_digits = str.maketrans('', '', string.digits)
    record_text = record_text.translate(remove_digits)
    record_text = record_text.translate(str.maketrans(punc_list, replacement))
    text_tokens = word_tokenize(record_text)
    tokens_without_sw = [porter_stemmer.stem(word.lower()) for word in text_tokens if
                         not word.lower() in stopwords.words('english')]
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
        record_text += " " + " ".join(record.xpath(".//TOPIC/text()"))
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
    tokens_without_sw = [porter_stemmer.stem(word.lower()) for word in text_tokens if
                         not word.lower() in stopwords.words('english')]
    max_count = 0
    for token in tokens_without_sw:
        if token in query_vector:
            query_vector[token] += 1
        else:
            query_vector[token] = 1
        if query_vector[token] > max_count:
            max_count = query_vector[token]
    return {k: v / max_count for k, v in query_vector.items()}


def print_relevant_documents(query, print_to_file=True):
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

    sorted_docs = [doc.lstrip("0") for doc in sorted(doc_to_score, key=lambda k: doc_to_score[k], reverse=True)]
    if print_to_file:
        write_file = open('ranked_query_docs.txt', 'w')
        for doc_num in sorted_docs:
            if doc_to_score[doc_num] > 0:
                write_file.write(doc_num + '\n')
        write_file.close()

    return sorted_docs[:45]


def calc_precision(docs_returned, relevant_docs):
    return len(docs_returned.intersection(relevant_docs)) / len(docs_returned)


def calc_recall(docs_returned, relevant_docs):
    return len(docs_returned.intersection(relevant_docs)) / len(relevant_docs)


def calc_acc_for_all_queries(query_path):
    count = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    root = etree.parse(query_path)
    for query in root.xpath(".//QUERY"):
        count += 1
        query_text = ''.join(query.xpath(".//QueryText/text()"))
        items = query.xpath(".//Item/text()")
        relevant_docs = set([item.strip() for item in items])
        docs_returned = set(print_relevant_documents(query_text, False))
        precision = calc_precision(docs_returned, relevant_docs)
        recall = calc_recall(docs_returned, relevant_docs)
        if recall == 0 or precision == 0:
            total_f1 += 0
        else:
            total_f1 += 2 / ((1 / recall) + (1 / precision))
        total_recall += recall
        total_precision += precision

    return total_precision / count, total_recall / count, total_f1 / count


def calculate_acc(inverted_index_path="vsm_inverted_index.json", query_path="cfc-xml_corrected/cfquery.xml"):
    parse_inverted_index(inverted_index_path)
    precision, recall, f1 = calc_acc_for_all_queries(query_path)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)


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
    # main()
    calculate_acc()
