import random
import json
import os
import regex as re
import csv
from bs4 import BeautifulSoup
import jsonlines

def create_refs(category):
    currentpath = os.getcwd()

    allpredictions = []

    with jsonlines.open(currentpath + '/Data/'+ category + '/' + category + '_all_predictions.json') as reader:
        for obj in reader:
            allpredictions.append(obj)

    with open(currentpath + '/Data/'+ category + '/' + category + '_all_predictions_indices.json', 'r') as infile:
        allpredictionindices = json.load(infile)

    testpartlist = ['dev', 'test', '125', '250', '500', '1000']

    for testpart in testpartlist:
        testpartpredictions = allpredictions[allpredictionindices[testpart][0]:allpredictionindices[testpart][1]]
        testpartpredictionsrefs = []
        for testpartprediction in testpartpredictions:
            testpartpredictionsrefs.append(testpartprediction['translation']['data'])

        testpartpredictionsrefsstring = '\n'.join(testpartpredictionsrefs)
        if not os.path.isdir(currentpath + '/Predictions/' + category):
            os.mkdir(currentpath + '/Predictions/' + category)
        with open(currentpath + '/Predictions/' + category + '/' + category + '_' + testpart + '_refs.txt', 'w') as rf:
            rf.write(testpartpredictionsrefsstring)

def create_hypos(category):
    currentpath = os.getcwd()
    with open(currentpath + '/Model/' + category + '-translation/generated_predictions.txt') as hf:
        allpredictions = hf.readlines()

    with open(currentpath + '/Data/'+ category + '/' + category + '_all_predictions_indices.json', 'r') as infile:
        allpredictionindices = json.load(infile)

    testpartlist = ['dev', 'test', '125', '250', '500', '1000']

    for testpart in testpartlist:
        testpartpredictions = allpredictions[allpredictionindices[testpart][0]:allpredictionindices[testpart][1]]
        testpartpredictionsrefsstring = ''.join(testpartpredictions)
        if not os.path.isdir(currentpath + '/Predictions/' + category):
            os.mkdir(currentpath + '/Predictions/' + category)
        with open(currentpath + '/Predictions/' + category + '/' + category + '_' + testpart + '_hypos.txt', 'w') as hf:
            hf.write(testpartpredictionsrefsstring)

def multi_f1_command(category):
    currentpath = os.getcwd()

    testpartlist = ['dev', 'test', '125', '250', '500', '1000']

    for testpart in testpartlist:
        print(category, testpart)
        os.system('python3 ' + currentpath + '/multi-f1.py ' + currentpath + '/Predictions/' + category + '/' + category + '_' + testpart + '_hypos.txt ' + currentpath + '/Predictions/' + category + '/' + category + '_' + testpart + '_refs.txt')

categorylist = ['Airport', 'Astronaut', 'Building', 'City', 'ComicsCharacter', 'Food', 'Monument', 'SportsTeam', 'University', 'WrittenWork']
for category in categorylist:
    print(category)
    create_refs(category)
    create_hypos(category)
    multi_f1_command(category)



