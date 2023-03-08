from bs4 import BeautifulSoup
import spacy
import os
#nlp = spacy.load("nl_core_news_lg")
nlp = spacy.load("en_core_web_lg")
import regex as re
from nltk import ngrams
import pickle
import json
from augment.replace import BertSampler
from sacremoses import MosesDetokenizer
md = MosesDetokenizer(lang='en')

def position_of_ngram(words, hyp):
    length = len(words)
    for i, sublist in enumerate((hyp[i:i + length] for i in range(len(hyp)))):
        if words == sublist:
            return i, i+length
    return None, None

def wordtokenizer(text):
    nlptext = nlp(text)
    #Tokenize the text using SpaCy
    tokenlist = [token.text for token in nlptext if token.text != ' ']

    return tokenlist

def main():
    #Gather all E2E files
    filelist = []

    for path, subdirs, files in os.walk('C:/Users/cvdrl/Desktop/CACAPO/en/'):
        for name in files:
            if name.endswith('.xml'):
                filelist.append(os.path.join(path, name))

    allentrytemplateinfo = {}

    #filelist = ['C:/Users/cvdrl/Desktop/test.xml']

    currentpath = os.getcwd()

    for webnlgfile in filelist:
        #Open the file, gather all entries, and all lexicalizations for that entry, then also find the template and text for that lexicalization
        with open(webnlgfile, 'rb') as f:
            soup = BeautifulSoup(f, 'lxml')
        entrylist = soup.find('entries').find_all('entry')
        fileentrytemplateinfo = []
        for entry in entrylist:
            targetlist = entry.find_all('lex')
            entrytemplatelist = []
            for target in targetlist:
                #targettext = target.find('text').text
                targettemplate = target.find('template').text
                #The template is already easily tokenizable
                tokentargettemplate = targettemplate.split()

                targettemplatedict = {'eid': entry['eid'], 'lid': target['lid'], 'text': '', 'template': targettemplate, 'text_tokenized': '', 'template_tokenized': tokentargettemplate, 'info': [], 'references': []}

                data = target.find('sortedtripleset').find_all('striple')
                datalist = []
                for d in data:
                    datalist.append(d.text)

                targettemplatedict.update({'data': datalist})

                tokentexttemplate = tokentargettemplate.copy()

                referencelist = target.find('references').find_all('reference')

                #Iterate over the target text until the word index overlaps with a template indicator in the template text
                newwordidx = 0
                for wordidx, word in enumerate(tokentargettemplate):
                    if re.search(r'(((AGENT)|(PATIENT))\-\d+)', tokentargettemplate[wordidx]):
                        tag = re.search(r'(((AGENT)|(PATIENT)|(BRIDGE))\-\d+)', tokentargettemplate[wordidx]).group(1)
                        templatedict = {'tag': tag, 'wordmatches': '', 'indices': ''}

                        for ref in referencelist:
                            if tag == ref['tag']:
                                templatedict['wordmatches'] = wordtokenizer(ref.text)
                                templatedict['indices'] = list(range(newwordidx, newwordidx + len(wordtokenizer(ref.text))))
                                newwordidx += (len(wordtokenizer(ref.text))-1)
                                tokentexttemplate[wordidx] = wordtokenizer(ref.text)

                        targettemplatedict['info'].append(templatedict)

                    newwordidx += 1

                tokentexttemplate = [[x] if not isinstance(x, list) else x for x in tokentexttemplate]
                tokentexttemplate = [item for sublist in tokentexttemplate for item in sublist]
                #for inf in targettemplatedict['info']:
                    #print(tokentexttemplate[inf['indices'][0]:inf['indices'][-1]+1])

                if targettemplatedict['text_tokenized'] == '':
                    targettemplatedict['text_tokenized'] = tokentexttemplate
                if targettemplatedict['text'] == '':
                    #texttemplate = md.detokenize(tokentexttemplate)
                    texttemplate = ' '.join(tokentexttemplate)
                    targettemplatedict['text'] = texttemplate

                for ref in referencelist:
                    targettemplatedict['references'].append({'entity': ref['entity'], 'number': ref['number'], 'tag': ref['tag'], 'type': ref['type'], 'text': ref.text})

                entrytemplatelist.append(targettemplatedict)
            fileentrytemplateinfo.append(entrytemplatelist)
        allentrytemplateinfo.update({webnlgfile: fileentrytemplateinfo})

    with open(currentpath + '/Data/AllEntryTemplateInfo_CACAPO.json', 'w') as outfile:
        json.dump(allentrytemplateinfo, outfile, indent=4, separators=(',', ': '))

def convert_data(candidate, targettemplatedict, idxlist):
    tokenizedcandidate = candidate.split()
    newtargettemplatedict = targettemplatedict.copy()
    newtargettemplatedict['data'] = [x.split(' | ') for x in newtargettemplatedict['data']]

    datadict = {}
    for output_element in newtargettemplatedict['info']:
        for idx in idxlist:
            if ('indices' in output_element) and (idx in output_element['indices']):
                newmatch = tokenizedcandidate[idx]
                oldmatchidx = output_element['indices'].index(idx)
                oldmatch = output_element['wordmatches'][oldmatchidx]
                #Use the tag to find the corresponding data element
                for refidx, refval in enumerate(newtargettemplatedict['references']):
                    if refval['tag'] == output_element['tag']:
                        oldentity = refval['entity']
                        newtargettemplatedict['references'][refidx]['entity'] = newtargettemplatedict['references'][refidx]['entity'].replace(oldmatch, newmatch)

                        #Now we have the entities that we want to replace, so let's search for these entities in the data
                        for dataidx, dataval in enumerate(newtargettemplatedict['data']):
                            if dataval[1] == oldentity:
                                newtargettemplatedict['data'][dataidx][1] = newtargettemplatedict['references'][refidx]['entity']

        datadict = {}

        for datapoint in newtargettemplatedict['data']:
            if datapoint[0] not in datadict:
                datadict.update({datapoint[0]: [datapoint[1]]})
            else:
                datadict[datapoint[0]].append(datapoint[1])

    datalist = []

    for entry in datadict:
        if len(datadict[entry]) == 1:
            datalist.append(entry.upper() + '(' + entry + '="' + datadict[entry][0] + '")')
        else:
            dslist = []
            for occurrenceidx, occurrence in enumerate(datadict[entry]):
                dslist.append(entry + str(occurrenceidx + 1) + '="' + occurrence + '"')
            dsstring = ','.join(dslist)
            datastring = entry.upper() + '(' + dsstring + ')'
            datalist.append(datastring)

    datastring = ' '.join(datalist)

    return datastring

def data_augmentation(allentrytemplateinfo, domain):
    candidates125list = []
    candidates250list = []
    candidates500list = []
    candidates1000list = []

    rep = BertSampler(sim_threshold=0.001)
    currentpath = os.getcwd()

    if os.path.isfile(currentpath + '/DonePickle_CACAPO.pkl'):
        previousdonelist = []
        with open(currentpath + '/DonePickle_CACAPO.pkl', 'rb') as fr:
            try:
                while True:
                    previousdonelist.append(pickle.load(fr))
            except EOFError:
                pass
        startsearch = 'y'
    else:
        startsearch = 'n'

    for entrytemplateidx, entrytemplate in enumerate(allentrytemplateinfo):
        for targettemplatedictidx, targettemplatedict in enumerate(entrytemplate):
            if startsearch == 'y':
                entryfound = 'n'
                for prevdone in previousdonelist:
                    if (entrytemplateidx == prevdone['entrytemplateidx']) and (targettemplatedictidx == prevdone['targettemplatedictidx']):
                        entryfound = 'y'
                        break
                if entryfound == 'y':
                    continue
                else:
                    startsearch = 'n'

            try:
                doc = nlp(targettemplatedict['text'])
            except IndexError:
                continue
            idxlist = []
            for idx, token in enumerate(doc):
                #print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)
                if ((token.tag_.startswith('NN')) and (token.dep_ != 'compound')) or (token.pos_ == 'ADJ') or (token.pos_ == 'ADV'): #or (token.pos_ == 'VERB'):
                    idxlist.append(idx)

            #candidateslist = [o for o in rep(targettemplatedict['info'][0]['text'], idxlist, 20, dropout=0.2)]
            try:
                candidateslist = [o for o in rep(targettemplatedict['text'], idxlist, 20, dropout=0.2)]
                print(candidateslist, flush=True)
            except TypeError:
                continue

            with open(currentpath + '/DonePickle_CACAPO.pkl', 'ab') as f:
                pickle.dump({'entrytemplateidx': entrytemplateidx, 'targettemplatedictidx': targettemplatedictidx, 'candidateslist': candidateslist, 'targettemplatedict': targettemplatedict, 'idxlist': idxlist}, f)

            #allentrytemplateinfo[entrytemplateidx][targettemplatedictidx].update({'candidateslist': candidateslist})

            for candidateidx, candidate in enumerate(candidateslist):
                try:
                    candidatedatastring = convert_data(candidate, targettemplatedict, idxlist)
                except TypeError:
                    continue

                candidatedatastring = re.sub(r'\n', ' ', candidatedatastring)
                candidatetxt = re.sub(r'\n', ' ', candidate)
                candidatetxt = wordtokenizer(candidatetxt)
                candidatetxt = md.detokenize(candidatetxt)

                if candidateidx < 1:
                    candidates125list.append([candidatetxt, candidatedatastring])
                    candidates250list.append([candidatetxt, candidatedatastring])
                    candidates500list.append([candidatetxt, candidatedatastring])
                    candidates1000list.append([candidatetxt, candidatedatastring])
                elif candidateidx < 2:
                    candidates250list.append([candidatetxt, candidatedatastring])
                    candidates500list.append([candidatetxt, candidatedatastring])
                    candidates1000list.append([candidatetxt, candidatedatastring])
                elif candidateidx < 5:
                    candidates500list.append([candidatetxt, candidatedatastring])
                    candidates1000list.append([candidatetxt, candidatedatastring])
                elif candidateidx < 10:
                    candidates1000list.append([candidatetxt, candidatedatastring])
                else:
                    break

    rep = None  # NOTE: clear out GPU memory

    currentpath = os.getcwd()

    candidatesdict = {'125': candidates125list, '250': candidates250list, '500': candidates500list, '1000': candidates1000list}

    for candlist in candidatesdict:
        candidatestrg = [x[0] for x in candidatesdict[candlist]]
        candidatessrc = [x[1] for x in candidatesdict[candlist]]

        alltrgstring = '\n'.join(candidatestrg)
        allsrcstring = '\n'.join(candidatessrc)

        with open(currentpath + '/Predictions/CACAPO/Extended' + candlist + '_' + domain + '_trg.txt', 'wb') as f:
            f.write(bytes(alltrgstring, 'UTF-8'))

        with open(currentpath + '/Predictions/CACAPO/Extended' + candlist + '_' + domain + '_src.txt', 'wb') as f:
            f.write(bytes(allsrcstring, 'UTF-8'))

def collect_dict():
    currentpath = os.getcwd()

    with open(currentpath + '/Data/AllEntryTemplateInfo_CACAPO.json', 'r') as infile:
        allentrytemplateinfo = json.load(infile)

    fulltrain = []

    for e2efile in allentrytemplateinfo:
        if 'Train' in e2efile:
            if 'Incidents' in e2efile:
                data_augmentation(allentrytemplateinfo[e2efile], 'Incidents')
            elif 'Sports' in e2efile:
                data_augmentation(allentrytemplateinfo[e2efile], 'Sports')
            elif 'Stocks' in e2efile:
                data_augmentation(allentrytemplateinfo[e2efile], 'Stocks')
            elif 'Weather' in e2efile:
                data_augmentation(allentrytemplateinfo[e2efile], 'Weather')





#allentrytemplateinfo = main()
collect_dict()

#candidate = 'The head of the is Jacob Berg .'
#targettemplatedict = {'eid': 'Id1', 'lid': 'Id1', 'text': 'The leader of Aarhus is Jacob Bundsgaard .', 'template': 'The leader of AGENT-1 is PATIENT-1 .', 'text_tokenized': ['The', 'leader', 'of', 'Aarhus', 'is', 'Jacob', 'Bundsgaard', '.'], 'template_tokenized': ['The', 'leader', 'of', 'AGENT-1', 'is', 'PATIENT-1', '.'], 'info': [{'tag': 'AGENT-1', 'wordmatches': ['Aarhus'], 'indices': [3]}, {'tag': 'PATIENT-1', 'wordmatches': ['Jacob', 'Bundsgaard'], 'indices': [5, 6]}], 'references': [{'entity': 'Aarhus', 'number': '1', 'tag': 'AGENT-1', 'type': 'name', 'text': 'Aarhus'}, {'entity': 'Jacob_Bundsgaard', 'number': '2', 'tag': 'PATIENT-1', 'type': 'name', 'text': 'Jacob Bundsgaard'}], 'data': ['Aarhus | leaderName | Jacob_Bundsgaard']}
#idxlist = [1, 3, 6]
#candidatedatastring = convert_data(candidate, targettemplatedict, idxlist)

