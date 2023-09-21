import json
import os
import random
import re
from collections import OrderedDict

import addressparser
import datefinder
import dateparser
import nltk
import numpy as np
import pandas as pd
import pdfplumber
import pyap
import PyPDF2
import spacy
import spacy_transformers
import torch
import uuid

from dateparser.search import search_dates
from dateutil import parser
from flask import jsonify
from nameparser import HumanName
from nltk import ne_chunk, pos_tag, sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from pdfminer.high_level import extract_pages, extract_text, extract_text_to_fp
from sentence_transformers import SentenceTransformer, util
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    pipeline,
    set_seed,
)


class PDFTextExtrator:
    nlp = None
    nlp1 = None
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    model_name = "deepset/bert-large-uncased-whole-word-masking-squad2"

    def __init__(self):
        self.seed = 42
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        set_seed(42)

        if PDFTextExtrator.nlp is None:
            PDFTextExtrator.nlp = spacy.load("en_core_web_trf")

        if PDFTextExtrator.nlp1 is None:
            PDFTextExtrator.nlp1 = pipeline(
                "question-answering",
                model=PDFTextExtrator.model_name,
                tokenizer=PDFTextExtrator.model_name,
            )

    def extract_text(self, file, selected_options):
        file_path = str(uuid.uuid4()) + ".pdf"
        print("Saving temporary file for processing: ", file_path)
        try:
            file.save(file_path)
            with open(file_path, "rb") as f:
                text = extract_text(f)
                clean_text1 = text.split()
                # print(text)
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
                print("Removed temporary file that was processed: ", file_path)

        data = {}
        if "length_words" in selected_options:
            data["total_length_words"] = len(clean_text1)

        if "addresses" in selected_options or "owner_address" in selected_options or "tenant_address" in selected_options:
            symbols_to_remove = ['▪', '■', '►', '✦', '◦', '✶', '❖', '✪', '➤', "•"]
            cleaned_text = text
            for symbol in symbols_to_remove:
                cleaned_text = cleaned_text.replace(symbol, '')
            address_pattern = (
                r'(?:(?:PO BOX|Po Box|P\.O\. BOX)\s+\d+\s*[•-]*\s*[A-Za-z\s,]+\s*,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?)|'
                r'PO BOX\s\d+\s•\s[A-Za-z\s]+\,\s[A-Z]{2}\s\d{5}|'
                r'PO Box\s\d+\s•\s[A-Za-z\s]+\,\s[A-Z]{2}\s\d{5}(?:-\d{4})?|'
                r'(?:PO BOX|Po Box|P.O. BOX)\s+\d+\s*[•-]*\s*[A-Za-z\s,]+\s*,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?|'
                r"\b\d+\s[\w\s.-]+,\s\w+\s\d+\b"
            )
            addresses1 = re.findall(address_pattern, text)
            addresse1 = pyap.parse(cleaned_text, country="US")
            addresse2 = pyap.parse(cleaned_text, country="GB")  # uk
            addresse3 = pyap.parse(cleaned_text, country="CA")  # canada
            addresses = addresse1 + addresses1
            filtered_addresses = [str(address) for address in addresses if
                                  re.search(r'\b\d{5}(?:-\d{4})?\b', str(address))]
            all_addresses = filtered_addresses
            if all_addresses:
                unique_addresses = list(set(all_addresses))
                address_list = unique_addresses
                if "addresses" in selected_options:
                    data['addresses'] = {f"address_{idx}": address for idx, address in enumerate(unique_addresses, start=1)}
                    data['address_count'] = len(unique_addresses)


        if "dates" in selected_options:
            date_pattern = (
                r"(?i)\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|"
                r"\d{1,2}(?:st|nd|rd|th)? \w+ \d{2,4}|"
                r"\d{1,2} \w+ \d{2,4}|"
                r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{2,4}|"
                r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?) \d{1,2}, \d{4}|"
                r"[a-zA-Z]{3} \d{1,2}, \d{4}|"
                r"[a-zA-Z]{3} \d{1,2},\d{4})\b"
            )

            matches = re.findall(date_pattern, text, flags=re.IGNORECASE)
            dates = [parser.parse(match, fuzzy=True) for match in matches]
            unique_dates = list(set(date.strftime("%Y-%m-%d") for date in dates))
            # Filter out dates with starting year "0"
            valid_dates = [date for date in unique_dates if not date.startswith("0")]
            ordered_dates_dict = OrderedDict()
            for idx, date in enumerate(valid_dates, start=1):
                ordered_dates_dict[f"date_{idx}"] = date
            data["dates"] = ordered_dates_dict
            data["date_count"] = len(ordered_dates_dict)

        if "names" in selected_options:
            format_text = text.split()
            clean_text = " ".join(format_text[0:11000])
            doc = PDFTextExtrator.nlp(clean_text)
            extracted_names = []
            extracted_organizations = []
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    extracted_names.append(ent.text)
                elif ent.label_ == "ORG":
                    extracted_organizations.append(ent.text)
            all_extracted = set(extracted_names + extracted_organizations)
            words_to_remove = ["TENANT","LANDLORD","RESIDENT","Tenant","Landlord","Resident","Landlords","Lessee",
                               "LESSEEE","Lessor"]
            filtered_names = [
                name for name in all_extracted if name not in words_to_remove
            ]
            name_length_threshold = 40
            filtered_names = [name for name in filtered_names
                if (len(name) <= name_length_threshold and len(name) > 2)]
            formatted_names = {
                f"name_{idx}": name for idx, name in enumerate(filtered_names, start=1)
            }
            data["names"] = formatted_names
            data["name_count"] = len(formatted_names)

        if "full_text" in selected_options:
            data["full_text"] = text

        def find_usd_amount(text):
            usd_pattern = r"\$\s*(\d+(\.\d+)?)"
            matches = re.findall(usd_pattern, text)
            usd_amounts = [float(match[0]) for match in matches]
            return usd_amounts

        if "monetary_amounts" in selected_options:
            usd_amounts = find_usd_amount(text)
            formatted_amounts = {
                f"amount_{idx}": amount
                for idx, amount in enumerate(usd_amounts, start=1)
            }
            data["monetary_amounts"] = formatted_amounts
            data["monetary_amount_count"] = len(formatted_amounts)

        if "tenant_name" in selected_options:
            question = "what's the name of tenant/resident?"
            data["tenant_name"] = self.extract_name_dates(question, text)

        landlord_name = []
        if "owner_name" in selected_options or "owner_address" in selected_options:
            question = "What is the name of the landlord/owner company in the lease agreement?"
            owner_names =self.extract_name_dates(question, text)
            landlord_name = str(owner_names)
            if "owner_name" in selected_options:
                data["owner_name"] = owner_names

        if "lease_start_date" in selected_options:
            question = "what is the lease start date?"
            data["lease_start_date"] = self.extract_name_dates(question, text)

        if "lease_end_date" in selected_options:
            question = "what is the lease end date?"
            data["lease_end_date"] = self.extract_name_dates(question, text)

        if "tenant_address" in selected_options:
            question = "where tenant/resident address located ?"
            data["tenant_address"] = self.extract_address(question, text, address_list)


        if "owner_address" in selected_options:
            question = "What is the address of" + str(landlord_name) + "/Landlord's Address"
            data["owner_address"] = self.extract_address(question, text, address_list)

        return jsonify(data)

    def extract_name_dates(self, question, text):
        text_list = text.split()
        clean_text = " ".join(text_list[0:1300])
        QA_input = {"question": question,"context": clean_text}
        res = PDFTextExtrator.nlp1(QA_input)
        return res["answer"]

    def extract_address(self,question, text, address_list):
        words_list = text.split()
        clean_text = " ".join(words_list[0:1500])
        QA_input = {"question": question, "context": clean_text}
        res = PDFTextExtrator.nlp1(QA_input)
        address = res["answer"]
        source_sentence = str(address)
        target_sentences = [s for s in address_list]
        source_embedding = PDFTextExtrator.model.encode(
            source_sentence, convert_to_tensor=True
        )
        target_embeddings = PDFTextExtrator.model.encode(
            target_sentences, convert_to_tensor=True
        )
        similarities = util.pytorch_cos_sim(source_embedding, target_embeddings)[0]
        highest_similarity_index = similarities.argmax()
        highest_similarity = similarities[highest_similarity_index]
        most_similar_target = target_sentences[highest_similarity_index]
        similarity_threshold = 0.3
        matched_address = []
        if highest_similarity >= similarity_threshold:
            matched_address = most_similar_target

        return matched_address

if __name__ == "__main__":
    extractor = PDFTextExtrator()
    print(extractor.extract_text())