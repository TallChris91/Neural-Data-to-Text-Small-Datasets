import random
import json
import os
import regex as re
import csv
from bs4 import BeautifulSoup
import jsonlines

def create_refs(language, category):
    currentpath = os.getcwd()

    allpredictions = []

    with jsonlines.open(currentpath + '/Data/' + language + '/' + category + '/' + category + '_all_predictions.json') as reader:
        for obj in reader:
            allpredictions.append(obj)

    with open(currentpath + '/Data/' + language + '/' + category + '/' + category + '_all_predictions_indices.json', 'r') as infile:
        allpredictionindices = json.load(infile)

    testpartlist = ['dev', 'test', '125', '250', '500', '1000']

    for testpart in testpartlist:
        testpartpredictions = allpredictions[allpredictionindices[testpart][0]:allpredictionindices[testpart][1]]
        testpartpredictionsrefs = []
        for testpartprediction in testpartpredictions:
            testpartpredictionsrefs.append(testpartprediction['translation']['data'])

        testpartpredictionsrefsstring = '\n'.join(testpartpredictionsrefs)

        os.makedirs(currentpath + '/Predictions/' + category + '/' + language, exist_ok=True)

        with open(currentpath + '/Predictions/' + category + '/' + language + '/' + language.upper() + '_' + category + '_' + testpart + '_refs.txt', 'w') as rf:
            rf.write(testpartpredictionsrefsstring)

def create_hypos(language, category):
    currentpath = os.getcwd()
    if language == 'en':
        with open(currentpath + '/Model/' + language.upper() + '-' + category + '-T5/generated_predictions.txt') as hf:
            allpredictions = hf.readlines()
    elif language == 'nl':
        with open(currentpath + '/Model/' + language.upper() + '-' + category + '-MT5/generated_predictions.txt') as hf:
            allpredictions = hf.readlines()

    allpredictions = [re.sub('^data\s', '', x) for x in allpredictions]

    with open(currentpath + '/Data/' + language + '/' + category + '/' + category + '_all_predictions_indices.json', 'r') as infile:
        allpredictionindices = json.load(infile)

    testpartlist = ['dev', 'test', '125', '250', '500', '1000']

    for testpart in testpartlist:
        testpartpredictions = allpredictions[allpredictionindices[testpart][0]:allpredictionindices[testpart][1]]
        testpartpredictionsrefsstring = ''.join(testpartpredictions)

        os.makedirs(currentpath + '/Predictions/' + category + '/' + language, exist_ok=True)

        with open(currentpath + '/Predictions/' + category + '/' + language + '/' + language.upper() + '_' + category + '_' + testpart + '_hypos.txt', 'w') as hf:
            hf.write(testpartpredictionsrefsstring)

def multi_f1_command(language, category):
    currentpath = os.getcwd()

    testpartlist = ['dev', 'test', '125', '250', '500', '1000']

    for testpart in testpartlist:
        print(category, language, testpart)
        os.system('python3 ' + currentpath + '/multi-f1.py ' + currentpath + '/Predictions/' + category + '/' + language + '/' + language.upper() + '_' + category + '_' + testpart + '_hypos.txt ' + currentpath + '/Predictions/' + category + '/' + language + '/' + language.upper() + '_' +  category + '_' + testpart + '_refs.txt')

languagelist = ['en', 'nl']
#languagelist = ['nl']
for language in languagelist:
    categorylist = ['Accidents', 'Sports', 'Stocks', 'Weather']
    for category in categorylist:
        create_refs(language, category)
        create_hypos(language, category)
        multi_f1_command(language, category)



