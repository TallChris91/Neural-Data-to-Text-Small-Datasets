import random
import json
import os
import regex as re
import csv
from bs4 import BeautifulSoup
import jsonlines

def modify_triples(datastring):
    datalist = datastring.split('], ')
    datalist = [x + ']' for x in datalist]
    datalist = [re.sub(r']]', ']', x) for x in datalist]

    newdatalist = []
    for element in datalist:
        keyvalue = re.search(r'^([^\[]*?)\[([^\]]*?)\]', element)
        newdatalist.append(keyvalue.group(1)+ ' @SEP@ ' + keyvalue.group(2))
    newdatalist = ' @EOF@ '.join(newdatalist)

    return newdatalist

def read_csv(inputfile):
    entrylist = []
    with open(inputfile, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        for idx, row in enumerate(reader):
            if idx > 0:
                entrylist.append(row)

    return entrylist

def collect_plus():
    currentpath = os.getcwd()
    allentries = []
    tdtlist = ['train', 'dev', 'test']
    for tdt in tdtlist:
        entries = read_csv(currentpath + '/Original/' + tdt + 'set_plus.csv')
        allentries += entries

    get125 = int(round(len(allentries) / 8, 0))
    get250 = int(round(len(allentries) / 4, 0))
    get500 = int(round(len(allentries) / 2, 0))

    #And a version that contains 500 files from those 1000
    sample500list = random.sample(allentries, get500)

    #And a version that contains 250 files from those 500
    sample250list = random.sample(sample500list, get250)

    #And a version that contains 125 files from those 250
    sample125list = random.sample(sample250list, get125)

    plusdict = {'1000': allentries, '500': sample500list, '250': sample250list, '125': sample125list}

    return plusdict


def read_xml(traindevtest):
    currentpath = os.getcwd()
    allentrylists = []

    if traindevtest == 'all_predictions':
        allentryindices = {'dev': [0], 'test': [], '125': [], '250': [], '500': [], '1000': []}
        all_predictions_set = ['dev', 'test', '125', '250', '500', '1000']
        plusdict = collect_plus()
        for predictionsetidx, predictionset in enumerate(all_predictions_set):
            if (predictionset == 'dev') or (predictionset == 'test'):
                entries = read_csv(currentpath + '/Original/' + predictionset + 'set.csv')
            else:
                entries = plusdict[predictionset]

            for idx, entry in enumerate(entries):
                datapart = modify_triples(entry[0].strip())
                graphdict = {'translation': {'en': entry[1].strip(), 'data': datapart.strip()}}

                allentrylists.append(graphdict)

            if predictionset != '1000':
                allentryindices[predictionset].append(len(allentrylists))
                allentryindices[all_predictions_set[predictionsetidx+1]].append(len(allentrylists))
            else:
                allentryindices[predictionset].append(len(allentrylists))

        os.makedirs(currentpath + '/Data/', exist_ok=True)

        with open(currentpath + '/Data/' + traindevtest + 'set_indices.json', 'w') as outfile:
            json.dump(allentryindices, outfile, indent=4, separators=(',', ': '))


    else:
        entries = read_csv(currentpath + '/Original/' + traindevtest + 'set.csv')

        for idx, entry in enumerate(entries):
            datapart = modify_triples(entry[0].strip())
            graphdict = {'translation': {'en': entry[1].strip(), 'data': datapart}}

            allentrylists.append(graphdict)

    os.makedirs(currentpath + '/Data/', exist_ok=True)

    with jsonlines.open(currentpath + '/Data/' + traindevtest + 'set.json', mode='w') as writer:
        writer.write_all(allentrylists)


traindevtestlist = ['train', 'dev', 'test', 'all_predictions']
for traindevtest in traindevtestlist:
    read_xml(traindevtest)