import random
import json
import os
import regex as re
import csv
from bs4 import BeautifulSoup
import jsonlines

def create_refs():
    currentpath = os.getcwd()

    allpredictions = []

    with jsonlines.open(currentpath + '/Data/' + 'all_predictionsset.json') as reader:
        for obj in reader:
            allpredictions.append(obj)

    with open(currentpath + '/Data/' + 'all_predictionsset_indices.json', 'r') as infile:
        allpredictionindices = json.load(infile)

    testpartlist = ['dev', 'test', '125', '250', '500', '1000']

    for testpart in testpartlist:
        testpartpredictions = allpredictions[allpredictionindices[testpart][0]:allpredictionindices[testpart][1]]
        testpartpredictionsrefs = []
        for testpartprediction in testpartpredictions:
            testpartpredictionsrefs.append(testpartprediction['translation']['data'])

        testpartpredictionsrefsstring = '\n'.join(testpartpredictionsrefs)

        os.makedirs(currentpath + '/Predictions/', exist_ok=True)

        with open(currentpath + '/Predictions/' + testpart + '_refs.txt', 'w') as rf:
            rf.write(testpartpredictionsrefsstring)

def create_hypos():
    currentpath = os.getcwd()
    with open(currentpath + '/Model/T5/generated_predictions.txt') as hf:
        allpredictions = hf.readlines()

    with open(currentpath + '/Data/' + 'all_predictionsset_indices.json', 'r') as infile:
        allpredictionindices = json.load(infile)

    testpartlist = ['dev', 'test', '125', '250', '500', '1000']

    for testpart in testpartlist:
        testpartpredictions = allpredictions[allpredictionindices[testpart][0]:allpredictionindices[testpart][1]]
        testpartpredictionsrefsstring = ''.join(testpartpredictions)

        os.makedirs(currentpath + '/Predictions/', exist_ok=True)

        with open(currentpath + '/Predictions/' + testpart + '_hypos.txt', 'w') as hf:
            hf.write(testpartpredictionsrefsstring)

def multi_f1_command():
    currentpath = os.getcwd()

    testpartlist = ['dev', 'test', '125', '250', '500', '1000']

    for testpart in testpartlist:
        print(testpart)
        os.system('python3 ' + currentpath + '/multi-f1.py ' + currentpath + '/Predictions/' + testpart + '_hypos.txt ' + currentpath + '/Predictions/' + testpart + '_refs.txt')


create_refs()
create_hypos()
multi_f1_command()



