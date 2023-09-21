from pdfminer.high_level import extract_text, extract_pages, extract_text_to_fp
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline,set_seed
from flask import Flask, request, render_template, jsonify
from sentence_transformers import SentenceTransformer, util
import json
import pyap
import PyPDF2
import pandas as pd
import numpy as np
from dateutil import parser
import datefinder
import dateparser
import nltk
import re
from nltk import sent_tokenize,word_tokenize, pos_tag
from nltk import ne_chunk
from nltk.corpus import stopwords
import pdfplumber
import spacy
from dateparser.search import search_dates
from nameparser import HumanName
import os
import addressparser
from collections import OrderedDict
app = Flask(__name__)
nlp = spacy.load('en_core_web_trf')
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
model_name = "deepset/bert-large-uncased-whole-word-masking-squad2"
nlp1 = pipeline('question-answering', model=model_name, tokenizer=model_name)
@app.route("/ml-service/health/v1/ping", methods=["GET"])
def home():
    if request.method == 'GET':
        return "pong"
@app.route("/ml-service/text-extraction", methods=["POST"])
def text_from_pdf():
    if request.method == 'POST':
        # print(request.files)
        file = request.files["file"]
        selected_options = request.form["extractOptions"]
        file_path = "temp.pdf"
        file.save(file_path)
        with open(file_path, 'rb') as f:
            text = extract_text(f)
            # print(text)
        data = {}
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
            date_pattern = r'(?i)\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|' \
                           r'\d{1,2}(?:st|nd|rd|th)? \w+ \d{2,4}|' \
                           r'\d{1,2} \w+ \d{2,4}|' \
                           r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{2,4}|' \
                           r'(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?) \d{1,2}, \d{4}|' \
                           r'[a-zA-Z]{3} \d{1,2}, \d{4}|' \
                           r'[a-zA-Z]{3} \d{1,2},\d{4})\b'

            matches = re.findall(date_pattern, text, flags=re.IGNORECASE)
            dates = [parser.parse(match, fuzzy=True) for match in matches]
            unique_dates = list(set(date.strftime("%Y-%m-%d") for date in dates))
            # Filter out dates with starting year "0"
            valid_dates = [date for date in unique_dates if not date.startswith("0")]
            ordered_dates_dict = OrderedDict()
            for idx, date in enumerate(valid_dates, start=1):
                ordered_dates_dict[f"date_{idx}"] = date
            data['dates'] = ordered_dates_dict
            data['date_count'] = len(ordered_dates_dict)

        if "names" in selected_options:
            format_text = text.split()
            clean_text = " ".join(format_text[0:6000])
            doc = nlp(clean_text)
            extracted_names = []
            extracted_organizations = []
            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    extracted_names.append(ent.text)
                elif ent.label_ == 'ORG':
                    extracted_organizations.append(ent.text)
            all_extracted = set(extracted_names + extracted_organizations)
            words_to_remove = ["Tenant", "Walk", "Please", "Emergency Service", "Lease","Front","HUD","Tenant", "Walk", "Please", "Emergency Service", "Lease","Front","Make","Emergency","LANDLORD","Rent Income","Rent","Landlord","Fire","X Giant Oaks, LLC",
                               "Gas","Lot","Lot Rent","Follows","Monthly","Resident","Fist","Items","Unit","Family Alliance","Landlord","Co-signer","Premises","the Term. Lessee","Tenant to Landlord","Tenant Portal","Deposit","Premises","the US",
                               "Buy Out", "Lessee", "Lessee. Lessee", "Lessor", "The Owner/Agent","the Owner/Agent","Lessor. Parking","Lessee","Lessor","the Armed Forces","the Lease Ending Date","Lessor. Parking","Tenant","Lease","X Giant Oaks,",
                               "TENANT","2 2.4 MAINTENANCE TENANT","2.8 Extermination. Tenant","2.9 Moisture","3.2 INSURANCE Landlord","Hours/ Emergencies","HOURS/ EMERGENCY CALLS Crandall Enterprises","Center Point","SCHUYLER 8.1 SCHUYLER UTILITIES LANDLORD",
                               "Tenant Portal","Tenant Portal","Tenant","Tenant. Landlord","Landlord.Lessee","Landlord to Tenant","Community Rules","Tenant. Tenant","Landlord","Landlords","Landlord and Resident","the Inventory Form","2018 Resident",
                               "NOON","Landlord","a Security Deposit","Clorox Cleanup","The Owner/Agent","the Lease Residents","Tenant’s Pet","2 Child/Children","Illinois Rd - Suite A","IN 46806 Tenant","German Shepherd. Landlord","DUST Tenant","Filters: Tenant",
                               "Tenant on Property. L. Plumbing","CARPETS","ETC. Tenant","[Tenant Initials] Tenant","X Landlord","Required Insurance","Lessor Insurance","Renters Insurance","the World Pet Registry","the Property. Lessee","Leasing/Maintenance",
                               "Lessees","the Property. Lessee","WHEREAS","the Leased Unit","the Leased Unit","Application","the Condition Form","SPECIAL ADDENDEUM FIREFLY & CHILDREN FAMILY ALLIANCE Firefly Children and Family Alliance","Lessee Daynesha D. Glover",
                               "any Resident Party","Landlord's Related Parties","Resident's","Admin Fee","Moveout Notice","Landlord Insurance","Move","Civil Relief Act","Savage Lessor IP Address","Lessee IP Address","Mail","Pool Key","Auto Pay","Pet Addendum",
                               "Renewal Term","Incorrect","Regulations Addendum","Titan","Landlord. Tenant","e Apartme","s Lea","State","Rodney Balser 3 Titan Management Group","State","Indiana Page","Ford","Fusion","Grey","Residents’","Keveoni Simmons Children",
                               "Tenants","2.5 UTILITIES Tenant","T- Tenant","L-Landlord","Landlord’s Office","Ashley L. Gaither 3","Property","Plaza","2 Furniture","Larry","Larry","Security Deposit","City of Kokomo","Co-Signor","8authorized","X Giant Oaks, LLC",
                               "Owner nor Management","EPA","Tenant Initials] Tenant","12 Floors","City","KCB","the Property. 7. Lessee","Lease Terms","LLC Lessor","Lease if Resident","8Apartment","Apartment.You","Clorox Cleanup®","the Leasing Office","Kara A.","Kara A",
                               "Mobile Home Owners","Tenant’s Apartment","Tenant’s Apartment","the Apartment Community","the Management Office","Ashley L. Gaither 4","the Tenant Portal","the Electric","Police","X Daynesha Glover","the Leasing Office","Lessor’s Disclosure",
                               "the Armed Forces of the United States","Center point","Force Majeure","Benjamin crossing","Water and Electric","Firefly Children and Family Alliance","KCB KCB Property Management","KCB Properties",
                               ]

            filtered_names = [name for name in all_extracted if name not in words_to_remove]
            name_length_threshold = 40
            filtered_names = [name for name in filtered_names if
                              (len(name) <= name_length_threshold and len(name) > 2)]
            formatted_names = {f"name_{idx}": name for idx, name in enumerate(filtered_names, start=1)}
            data['names'] = formatted_names
            data['name_count'] = len(formatted_names)

        if "full_text" in selected_options:
            data['full_text'] = text

        def find_usd_amount(text):
            usd_pattern = r'\$\s*(\d+(\.\d+)?)'
            matches = re.findall(usd_pattern, text)
            usd_amounts = [float(match[0]) for match in matches]
            return usd_amounts

        if "monetary_amounts" in selected_options:
            usd_amounts = find_usd_amount(text)
            formatted_amounts = {f"amount_{idx}": amount for idx, amount in enumerate(usd_amounts, start=1)}
            data['monetary_amounts'] = formatted_amounts
            data['monetary_amount_count'] = len(formatted_amounts)

        if "tenant_name" in selected_options:
            clean_text2 = text.split()
            c,d=0,1300
            first_200_words = ' '.join(clean_text2[c:d])
            QA_input1 = {
                'question': "what's the name of tenant/resident?",
                'context':first_200_words}
            res = nlp1(QA_input1)
            data["tenant_name"]=res['answer']

        landlord_name =[]
        if "owner_name" in selected_options or "owner_address" in selected_options :
            clean_text2 = text.split()
            e,f = 0,1300
            first_200_words = ' '.join(clean_text2[e:f])
            QA_input = {
                'question': "What is the name of the landlord/owner company in the lease agreement?",
                'context': first_200_words}
            res = nlp1(QA_input)
            landlord_name = str(res['answer'])
            if "owner_name" in selected_options:
                data["owner_name"] = res['answer']

        if "lease_start_date" in selected_options:
            clean_text2 = text.split()
            e, f = 0, 1200
            first_200_words = ' '.join(clean_text2[e:f])
            QA_input = {
                'question': "what is the lease start date?",
                'context': first_200_words}
            res = nlp1(QA_input)
            data["lease_start_date"] = res['answer']

        if "lease_end_date" in selected_options:
            clean_text2 = text.split()
            e, f = 0, 1200
            first_200_words = ' '.join(clean_text2[e:f])
            QA_input = {
                'question': "what is the lease end date?",
                'context': first_200_words}
            res = nlp1(QA_input)
            data["lease_end_date"] = res['answer']

        if "tenant_address" in selected_options:
            clean_text2 = text.split()
            e, f = 0, 1500
            first_200_words = ' '.join(clean_text2[e:f])
            QA_input = {
                'question': "where tenant/resident address located ?",
                'context': first_200_words}
            res = nlp1(QA_input)
            tenant_address = res['answer']
            source_sentence = str(tenant_address)
            target_sentences = [s for s in address_list]
            source_embedding = model.encode(source_sentence, convert_to_tensor=True)
            target_embeddings = model.encode(target_sentences, convert_to_tensor=True)
            similarities = util.pytorch_cos_sim(source_embedding, target_embeddings)[0]
            highest_similarity_index = similarities.argmax()
            highest_similarity = similarities[highest_similarity_index]
            most_similar_target = target_sentences[highest_similarity_index]
            similarity_threshold = 0.3
            c_list = []
            if highest_similarity >= similarity_threshold:
                # print(f"Highest Similarity: {highest_similarity:.4f}")
                # print("Most Similar Target Sentence:", most_similar_target)
                c_list = most_similar_target
            data["tenant_address"] = c_list

        if "owner_address" in selected_options:
            clean_text2 = text.split()
            e, f = 0, 1500
            first_200_words = ' '.join(clean_text2[e:f])
            question = "What is the address of" + str(landlord_name) + "/Landlord's Address"
            QA_input = {
                'question': question,
                'context': first_200_words}
            res = nlp1(QA_input)
            tenant_address = res['answer']
            source_sentence = str(tenant_address)
            target_sentences = [s for s in address_list]
            source_embedding = model.encode(source_sentence, convert_to_tensor=True)
            target_embeddings = model.encode(target_sentences, convert_to_tensor=True)
            similarities = util.pytorch_cos_sim(source_embedding, target_embeddings)[0]
            highest_similarity_index = similarities.argmax()
            highest_similarity = similarities[highest_similarity_index]
            most_similar_target = target_sentences[highest_similarity_index]
            similarity_threshold = 0.3
            c_list = []
            if highest_similarity >= similarity_threshold:
                c_list = most_similar_target
            data["owner_address"] = c_list

        os.remove(file_path)
        return jsonify(data)
    # http://localhost:5001/ml-service/text-extraction
    # https://india.yoroflow.com/ml-service/text-extraction
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)

