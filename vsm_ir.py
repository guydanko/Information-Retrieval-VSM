import os
import string
import sys
import xml.etree.ElementTree as ET
import lxml.html
from lxml import etree
import xml.etree.ElementTree as ETree
import nltk
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

# word : {file_number : how many times word in file_number}
inverted_index = {}
nltk.download('stopwords')
nltk.download('punkt')


def count_word_in_text(record_number, record_text):
    punc_list = '-.;:!?/\,#@$&)(\'"'
    replacement = ' ' * len(punc_list)
    remove_digits = str.maketrans('', '', string.digits)
    record_text = record_text.translate(remove_digits)
    record_text = record_text.translate(str.maketrans(punc_list, replacement))
    text_tokens = word_tokenize(record_text)
    tokens_without_sw = [word.lower() for word in text_tokens if not word in stopwords.words()]
    for token in tokens_without_sw:
        if token in inverted_index:
            record_dict = inverted_index[token]
            if record_number in record_dict:
                record_dict[record_number] += 1
            else:
                record_dict[record_number] = 1
        else:
            inverted_index[token] = {record_number: 1}


def parse_one_xml_file(doc):
    for record in doc.xpath(".//RECORD"):
        record_number = " ".join(record.xpath(".//RECORDNUM/text()"))
        record_text = ""
        record_text += " ".join(record.xpath(".//TITLE/text()"))
        record_text += " ".join(record.xpath(".//EXTRACT/text()"))
        record_text += " ".join(record.xpath(".//ABSTRACT/text()"))
        record_text += " ".join(record.xpath(".//TOPIC/text()"))
        count_word_in_text(record_number, record_text)

    print(inverted_index)


def build_inverted_index(path):
    for filename in os.listdir(path):
        root = etree.parse(path + "\\" + filename)
        parse_one_xml_file(root)


def main():
    if sys.argv[1] == "create_index":
        path = sys.argv[2]
        build_inverted_index(path)


if __name__ == '__main__':
    main()
