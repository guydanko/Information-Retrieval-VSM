import math
import os
import string
import sys
from lxml import etree
import json
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# word : {file_number : how many times word in file_number}
inverted_index = {}
stopwords = ['somehow', 'which', 'before', 'three', 'or', 'should', 'might', 'own', 'those', 'to', 'above', 'nor', 'me',
             'seems', 'after', 'empty', 'put', 'that', 'will', 'while', 'across', 'been', 'something', 'ie', 'from',
             'eight', 'herein', 'below', 'into', 'fifty', 'it', 'when', 'for', 'fifteen', 'top', 'hers', 'anyway',
             'between', 'nevertheless', 'the', 'still', 'whither', 'and', 'found', 'our', 'through', 'have',
             'whereupon', 'without', 'off', 'am', 'at', 'beside', 'four', 'himself', 'move', 'him', 'be', 'out', 'its',
             'thereupon', 'third', 'well', 'yet', 'such', 'themselves', 'as', 'thereafter', 'what', 'whoever',
             'sincere', 'until', 'too', 'many', 'not', 'whom', 'again', 'he', 'else', 'latter', 'of', 'on', 'anywhere',
             'towards', 'done', 'same', 'side', 'almost', 'find', 'upon', 'everything', 'hundred', 'often', 'thru',
             'twenty', 'are', 'afterwards', 'beforehand', 'bottom', 'except', 'ours', 'forty', 'rather', 'either',
             'meanwhile', 'since', 'then', 'thereby', 'because', 'once', 'whatever', 'wherein', 'you', 'do',
             'everywhere', 'during', 'front', 'she', 'detail', 'indeed', 'system', 'thin', 'name', 'his', 'others',
             'somewhere', 'now', 'whereafter', 'is', 'whereas', 'around', 'more', 'cannot', 'onto', 'seem', 'whole',
             'much', 'very', 'cry', 'hasnt', 'any', 'sometime', 'alone', 'etc', 'my', 'seeming', 'throughout', 'up',
             'their', 'anyone', 'can', 'yours', 'thus', 'take', 'nine', 'along', 'itself', 'ten', 'thence', 'there',
             'enough', 'further', 'go', 'interest', 'due', 'hereafter', 'few', 'back', 'formerly', 'here', 'nobody',
             'only', 'whenever', 'each', 'moreover', 'anyhow', 'how', 'also', 'un', 'amoungst', 'may', 'hereupon',
             'otherwise', 'us', 'was', 'give', 'over', 'some', 'under', 'than', 'becoming', 'amongst', 'mine', 'next',
             'fill', 'first', 'please', 'so', 'though', 'another', 'beyond', 'perhaps', 'see', 'fire', 'yourself',
             'none', 'whence', 'i', 'has', 'yourselves', 'full', 'noone', 'six', 'all', 'being', 'thick', 'least',
             'latterly', 'ltd', 'seemed', 'where', 'together', 'eg', 'other', 'show', 'whether', 'herself', 'among',
             'therefore', 'in', 'this', 'made', 'although', 'against', 'hereby', 'wherever', 'de', 'five', 'already',
             'could', 'two', 'your', 'never', 'eleven', 'most', 'sixty', 'a', 'however', 'one', 'but', 'her', 'if',
             'call', 'get', 'sometimes', 'twelve', 'within', 'mill', 'an', 'nowhere', 'must', 'con', 'everyone', 'per',
             'these', 'bill', 'keep', 'neither', 'myself', 'serious', 'we', 'whereby', 'nothing', 'always', 'amount',
             'becomes', 'namely', 'behind', 'last', 'mostly', 'therein', 'why', 'even', 'couldnt', 'ever', 'became',
             'every', 'down', 'about', 'elsewhere', 'ourselves', 'co', 're', 'by', 'who', 'via', 'former', 'several',
             'toward', 'both', 'would', 'someone', 'no', 'whose', 'less', 'describe', 'hence', 'anything', 'them',
             'cant', 'they', 'inc', 'part', 'had', 'become', 'were', 'besides', 'with']
stopwords_set = set(stopwords)
FREQ_KEY = 'FREQ'
TF_IDF_KEY = 'TF-IDF-SCORE'
KEY_FOR_AMOUNT_OF_DOCS = 'AMOUNT_OF_DOCS_IN_CORPUS'
KEY_FOR_WEIGHT_OF_DOCS = 'DOC_WEIGHT'
KEY_FOR_DOCUMENT_INFO = 'DOC_INFO'

porter_stemmer = PorterStemmer()

SCORE_THRESHOLD = 0.08


def count_word_in_text(record_number, record_text):
    """
    Update the inverted index for each record
       """
    record_number = record_number.lstrip("0").rstrip()
    # Clean all punctuation and digits from tokens
    punc_list = '-.;:!?/\,#@$&)(\'"'
    replacement = ' ' * len(punc_list)
    remove_digits = str.maketrans('', '', string.digits)
    record_text = record_text.translate(remove_digits)
    record_text = record_text.translate(str.maketrans(punc_list, replacement))
    text_tokens = word_tokenize(record_text)
    # Remove stopwords and stem tokens
    tokens_without_sw = [porter_stemmer.stem(word.lower()) for word in text_tokens if
                         not word.lower() in stopwords_set]
    # Update frequency of each word in inverted index
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
    """
       Extract all relevant text from a record
          """
    for record in doc.xpath(".//RECORD"):
        record_number = " ".join(record.xpath(".//RECORDNUM/text()"))
        record_text = ""
        record_text += " " + " ".join(record.xpath(".//TITLE/text()"))
        record_text += " " + " ".join(record.xpath(".//EXTRACT/text()"))
        record_text += " " + " ".join(record.xpath(".//ABSTRACT/text()"))
        record_text += " " + " ".join(record.xpath(".//TOPIC/text()"))
        count_word_in_text(record_number, record_text)


def update_tfidf_scores():
    """
       Calculate TF-IDF for each word in each document and update inverted index
          """
    doc_to_max_freq = {}
    doc_to_weights = {}
    # Calculate max frequency for each document
    for word in inverted_index:
        if word == KEY_FOR_DOCUMENT_INFO:
            continue
        for doc_num in inverted_index[word]:
            if doc_num in doc_to_max_freq:
                if inverted_index[word][doc_num][FREQ_KEY] > doc_to_max_freq[doc_num]:
                    doc_to_max_freq[doc_num] = inverted_index[word][doc_num][FREQ_KEY]
            else:
                doc_to_max_freq[doc_num] = inverted_index[word][doc_num][FREQ_KEY]
                doc_to_weights[doc_num] = 0
    # Update document info in inverted index
    num_documents = len(doc_to_max_freq)
    inverted_index[KEY_FOR_DOCUMENT_INFO] = {}
    inverted_index[KEY_FOR_DOCUMENT_INFO][KEY_FOR_AMOUNT_OF_DOCS] = num_documents
    inverted_index[KEY_FOR_DOCUMENT_INFO][KEY_FOR_WEIGHT_OF_DOCS] = doc_to_weights
    # Update scores to TF * IDF and total weight of each document vector
    for word in inverted_index:
        if word == KEY_FOR_DOCUMENT_INFO:
            continue
        for doc_num in inverted_index[word]:
            tf = inverted_index[word][doc_num][FREQ_KEY] / doc_to_max_freq[doc_num]
            idf = math.log2(num_documents / len(inverted_index[word]))
            tf_idf = tf * idf
            inverted_index[word][doc_num][TF_IDF_KEY] = tf_idf
            inverted_index[KEY_FOR_DOCUMENT_INFO][KEY_FOR_WEIGHT_OF_DOCS][doc_num] += tf_idf ** 2


def build_inverted_index(path):
    """
       Parse each xml file in given directory
          """
    for filename in os.listdir(path):
        if filename.endswith('.xml'):
            root = etree.parse(path + "\\" + filename)
            parse_one_xml_file(root)

    update_tfidf_scores()
    # Serializing json
    json_inverted_index = json.dumps(inverted_index)

    # Writing inverted index to file
    with open("vsm_inverted_index.json", "w") as outfile:
        outfile.write(json_inverted_index)
    outfile.close()


def parse_inverted_index(path):
    with open(path) as f:
        data = json.load(f)

    inverted_index.update(data)


def build_query_vector(query):
    """ Calculate query vector TF-IDF
       """
    query_vector = {}
    # Remove punctuation and digits
    punc_list = '-.;:!?/\,#@$&)(\'"'
    replacement = ' ' * len(punc_list)
    remove_digits = str.maketrans('', '', string.digits)
    query_text = query.translate(remove_digits)
    query_text = query_text.translate(str.maketrans(punc_list, replacement))
    text_tokens = word_tokenize(query_text)
    # Remove stopwords and stem tokens
    tokens_without_sw = [porter_stemmer.stem(word.lower()) for word in text_tokens if
                         not word.lower() in stopwords_set]
    max_count = 0
    # Count appearances of tokens
    for token in tokens_without_sw:
        if token in query_vector:
            query_vector[token] += 1
        else:
            query_vector[token] = 1
        if query_vector[token] > max_count:
            max_count = query_vector[token]
    weighted_query = {}
    # Calculate TF-IFD for each token in query
    for word in query_vector:
        tf = query_vector[word] / max_count
        if word in inverted_index:
            idf = math.log2(
                (inverted_index[KEY_FOR_DOCUMENT_INFO][KEY_FOR_AMOUNT_OF_DOCS] + 1) / len(inverted_index[word]))
        else:
            idf = math.log2(
                inverted_index[KEY_FOR_DOCUMENT_INFO][KEY_FOR_AMOUNT_OF_DOCS] + 1)
        weighted_query[word] = tf * idf
    return weighted_query


def print_relevant_documents(query, print_to_file=True):
    """ Retrieve relevant documents from VSM based on query
        print_to_file: If true, results will be exported to ranked_query_docs.txt
       """
    # Get the vector of the query
    query_vector = build_query_vector(query)
    query_vector_weight = 0
    doc_to_score = {}
    # Calculate cosine similarity of query to relevant documents and calculate query vector total weight
    for word in query_vector:
        query_vector_weight += query_vector[word] ** 2
        if word in inverted_index:
            for doc_num in inverted_index[word]:
                if doc_num in doc_to_score:
                    doc_to_score[doc_num] += inverted_index[word][doc_num][TF_IDF_KEY] * query_vector[word]
                else:
                    doc_to_score[doc_num] = inverted_index[word][doc_num][TF_IDF_KEY] * query_vector[word]
    # Normalize cosine similarity
    for doc_num in doc_to_score:
        doc_to_score[doc_num] = doc_to_score[doc_num] / (
            math.sqrt(query_vector_weight * inverted_index[KEY_FOR_DOCUMENT_INFO][KEY_FOR_WEIGHT_OF_DOCS][doc_num]))
    # Filter documents based on score (return only documents with score above the set threshold)
    sorted_docs = [doc.lstrip("0") for doc in sorted(doc_to_score, key=lambda k: doc_to_score[k], reverse=True) if
                   doc_to_score[doc] > SCORE_THRESHOLD]
    # Export results to file
    if print_to_file:
        write_file = open('ranked_query_docs.txt', 'w')
        for doc_num in sorted_docs:
            write_file.write(doc_num + '\n')
        write_file.close()

    return sorted_docs


# Functions for evaluation of VSM

def calc_precision(docs_returned, relevant_docs):
    """
       Calculate the precision for specific query
          """
    if len(docs_returned) == 0:
        return 0
    return len(docs_returned.intersection(relevant_docs)) / len(docs_returned)


def calc_recall(docs_returned, relevant_docs):
    """
        Calculate the recall for specific query
            """
    if len(docs_returned) == 0:
        return 0
    return len(docs_returned.intersection(relevant_docs)) / len(relevant_docs)


def calc_NDCG(docs_returned, items_to_scores):
    """
        Calculate the NDCG for specific query
            """
    items_to_gain = {}
    for item, score in items_to_scores.items():
        gain = sum(int(rating) for rating in score) / len(score)
        items_to_gain[item] = gain

    sorted_items = sorted(items_to_gain, key=lambda k: items_to_gain[k], reverse=True)
    idcg = items_to_gain[sorted_items[0]]
    for i in range(1, len(sorted_items)):
        idcg += items_to_gain[sorted_items[i]] / math.log2(i + 1)

    if len(docs_returned) == 0:
        return 0

    dcg = items_to_gain.get(docs_returned[0], 0)
    for i in range(1, len(docs_returned)):
        dcg += items_to_gain.get(docs_returned[i], 0) / math.log2(i + 1)

    return dcg / idcg


def calc_acc_for_all_queries(query_path):
    """
        Calculate metrics for all queries
        F1, NDCG, Precision, Recall
            """
    count = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_ndcg = 0
    root = etree.parse(query_path)
    for query in root.xpath(".//QUERY"):
        count += 1
        query_text = ''.join(query.xpath(".//QueryText/text()"))
        items = query.xpath(".//Item/text()")
        scores = query.xpath(".//Item/@score")
        items_to_scores = dict(zip(items, scores))
        relevant_docs = set([item.strip() for item in items])
        docs_returned = print_relevant_documents(query_text, False)
        docs_returned_set = set(docs_returned)
        total_ndcg += calc_NDCG(docs_returned, items_to_scores)
        precision = calc_precision(docs_returned_set, relevant_docs)
        recall = calc_recall(docs_returned_set, relevant_docs)
        try:
            total_f1 += (2 * precision * recall) / (precision + recall)
        except ZeroDivisionError:
            pass
        total_recall += recall
        total_precision += precision

    return total_precision / count, total_recall / count, total_f1 / count, total_ndcg / count


def calculate_acc(inverted_index_path="vsm_inverted_index.json", query_path="not_xml_files/cfquery.xml"):
    """
        Print all metrics for queries
            """
    parse_inverted_index(inverted_index_path)
    precision, recall, f1, ndcg = calc_acc_for_all_queries(query_path)
    print("Precision: {:0.3f}".format(precision))
    print("Recall: {:0.3f}".format(recall))
    print("F1: {:0.3f}".format(f1))
    print("NDCG: {:0.3f}".format(ndcg))
    return f1


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
    calculate_acc()
    main()
