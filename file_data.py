import os
import nltk

nltk.download("punkt")
import csv
from clean_data import clean_lines


def read_file(file_contents, tag):
    file_data = nltk.sent_tokenize(file_contents)

    # Writing the sentences to a CSV file
    with open(f"../fetched_data/{tag}_dataset.csv", mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows([[clean_lines(value)] for value in file_data])

    return 0
