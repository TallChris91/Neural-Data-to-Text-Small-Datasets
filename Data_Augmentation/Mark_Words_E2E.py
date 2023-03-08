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
    for idx, token in enumerate(tokenlist):
        #SpaCy struggles with the E2E templates (e.g., __NAME__ ). As it tends to make all underscores separate tokens. Let's fix that.
        #First, we find the start of the template
        try:
            if (tokenlist[idx] == '_') and (tokenlist[idx+1] == '_'):
                wordgroup = tokenlist[idx]
                dellist = []
                nextidx = idx
                #Go to the next words after the start of the template until you reach the end (market by two underscores and a non-underscore word).
                #Then we will group all the separate tokens into the one template token, and use the collected index information to delete the part-of-template tokens.
                while True:
                    nextidx += 1
                    try:
                        if (nextidx+2 == len(tokenlist)) or ((tokenlist[nextidx] == '_') and (tokenlist[nextidx+1] == '_') and (tokenlist[nextidx+2] != '_')):
                            dellist = dellist + [nextidx, nextidx+1]
                            wordgroup += tokenlist[nextidx] + tokenlist[nextidx+1]
                            break
                        else:
                            dellist.append(nextidx)
                            wordgroup += tokenlist[nextidx]
                    except IndexError:
                        return ['ERROR ERROR']

                #We reverse the indexlist to make sure the deletion doesn't affect the next index
                tokenlist[idx] = wordgroup
                for delnum in dellist:
                    tokenlist[delnum] = ''
        except IndexError:
            return ['ERROR ERROR']

    tokenlist = [x for x in tokenlist if x != '']

    return tokenlist

def main():
    #Gather all E2E files
    filelist = []

    for path, subdirs, files in os.walk('C:/Users/cvdrl/Desktop/EnrichedE2E-main'):
        for name in files:
            if name.endswith('.xml'):
                filelist.append(os.path.join(path, name))

    allentrytemplateinfo = {}

    currentpath = os.getcwd()

    for e2efile in filelist:
        #Open the file, gather all entries, and all lexicalizations for that entry, then also find the template and text for that lexicalization
        with open(e2efile, 'rb') as f:
            soup = BeautifulSoup(f, 'lxml')
        entrylist = soup.find('entries').find_all('entry')
        fileentrytemplateinfo = []
        for entry in entrylist:
            targetlist = entry.find_all('target')
            entrytemplatelist = []
            for target in targetlist:
                targettext = target.find('text').text
                targettemplate = target.find('template').text
                #Tokenize the targettext and template the same way as will be done in the Data_Augmentation file
                tokentargettext = wordtokenizer(targettext)
                targettext = ' '.join(tokentargettext)
                tokentargettemplate = wordtokenizer(targettemplate)
                if (tokentargettemplate == ['ERROR ERROR']) or (tokentargettext == ['ERROR ERROR']):
                    continue

                targettemplatedict = {'eid': entry['eid'], 'lid': target['lid'], 'info': []}

                templateissue = 'n'

                #Iterate over the target text until the word index overlaps with a template indicator in the template text
                for wordidx, word in enumerate(tokentargettext):
                    try:
                        if re.search(r'(__[A-Z]+_?[A-Z]+?__)', tokentargettemplate[wordidx]):
                            templatedict = {'tag': re.search(r'(__[A-Z]+_?[A-Z]+?__)', tokentargettemplate[wordidx]).group(1), 'wordmatches': [tokentargettext[wordidx]], 'indices': [wordidx], 'text': targettext, 'template': targettemplate, 'text_tokenized': tokentargettext, 'template_tokenized': tokentargettemplate}
                            nextlist = tokentargettext[wordidx+1:].copy()

                            for nextwordidx, nextword in enumerate(nextlist):
                                #If there is no next word in the template text anymore, add all remaining words to the dict.
                                if wordidx + 1 >= len(tokentargettemplate):
                                    templatedict['wordmatches'].append(nextword)
                                    templatedict['indices'].append(wordidx+1 + nextwordidx)
                                #Else stop if the next template word is found.
                                elif nextword == tokentargettemplate[wordidx+1]:
                                    break
                                else:
                                    templatedict['wordmatches'].append(nextword)
                                    templatedict['indices'].append(wordidx+1 + nextwordidx)

                            targettemplatedict['info'].append(templatedict)

                            matchindices = templatedict['indices'].copy()
                            if len(matchindices) > 1:
                                matchindices = matchindices[1:]
                                for matchidx in matchindices:
                                    tokentargettemplate.insert(matchidx, '_FILLER_')
                    except IndexError:
                        #print(tokentargettemplate)
                        #print(tokentargettext)
                        #print(targettext)
                        #print(e2efile)
                        #exit(2)
                        templateissue = 'y'

                if templateissue == 'y':
                    continue

                #ADD INFORMATION IF THE TEXT OVERLAPS WITH THE DATA AND WHERE IT OVERLAPS, SO THAT WE CAN CHANGE THIS WITH THE DATA AUGMENTATION
                data_inputlist = entry.find('source').find_all('input')
                for data_input in data_inputlist:
                    #TRY TO FIND N-GRAM MATCHES FOR MAX, THEN FOR MAX-1, MAX-2, etc.
                    #Iterate over the template info we collected
                    for idx, template_input in enumerate(targettemplatedict['info']):
                        #If the template_tag matches the data tag, let's see if there's overlapping text
                        if template_input['tag'] == data_input['tag']:
                            targettemplatedict['info'][idx].update({'data': {'attribute': data_input['attribute'], 'tag': data_input['tag'], 'value': data_input['value']}})
                            lexlist = template_input['indices'].copy()
                            ngramrange = list(range(len(lexlist), 0, -1))
                            ngramfound = 'n'
                            for ngramlen in ngramrange:
                                if ngramfound == 'n':
                                    lexngramspositions = list(ngrams(lexlist, ngramlen))
                                    lexngramspositions = [list(x) for x in lexngramspositions]
                                    for lexngram in lexngramspositions:
                                        wordmatchstart, wordmatchend = position_of_ngram(lexngram, lexlist)
                                        wordmatchinput = template_input['wordmatches'][wordmatchstart:wordmatchend]
                                        tokeninput = wordtokenizer(data_input['value'])
                                        startposition, endposition = position_of_ngram(wordmatchinput, tokeninput)
                                        if startposition != None:
                                            ngramfound = 'y'
                                            targettemplatedict['info'][idx].update({'overlap': lexngram})
                                            break
                                if ngramfound == 'y':
                                    break



                #print(targettemplatedict)
                entrytemplatelist.append(targettemplatedict)
            fileentrytemplateinfo.append(entrytemplatelist)
        allentrytemplateinfo.update({e2efile: fileentrytemplateinfo})

    with open(currentpath + '/Data/AllEntryTemplateInfo.json', 'w') as outfile:
        json.dump(allentrytemplateinfo, outfile, indent=4, separators=(',', ': '))

def convert_data(candidate, targettemplatedict, idxlist):
    tokenizedcandidate = wordtokenizer(candidate)
    l = [1, 2, 3]
    datadict = {}
    for output_element in targettemplatedict['info']:
        replaceindices = []
        for idx in idxlist:
            if ('overlap' in output_element) and (idx in output_element['overlap']):
                replaceindices.append([output_element['overlap'].index(idx), idx])
        try:
            datavalue = output_element['data']['value']
        except KeyError:
            print('ERROR ERROR', flush=True)
            return 'ERROR ERROR'
        datavaluelist = wordtokenizer(datavalue)
        for replaceidx in replaceindices:
            datavaluelist[replaceidx[0]] = tokenizedcandidate[replaceidx[1]]

        datavaluestring = md.detokenize(datavaluelist)
        datadict.update({output_element['data']['attribute']: datavaluestring})

    datalist = []

    for entry in datadict:
        datalist.append(entry.upper() + '(' + entry + '="' + datadict[entry] + '")')

    datastring = ' '.join(datalist)

    return datastring

def data_augmentation(allentrytemplateinfo):
    candidates125list = []
    candidates250list = []
    candidates500list = []
    candidates1000list = []

    rep = BertSampler(sim_threshold=0.001)
    currentpath = os.getcwd()


    if os.path.isfile(currentpath + '/DonePickle.pkl'):
        previousdonelist = []
        with open(currentpath + '/DonePickle.pkl', 'rb') as fr:
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
                doc = nlp(targettemplatedict['info'][0]['text'])
            except IndexError:
                continue
            idxlist = []
            for idx, token in enumerate(doc):
                #print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)
                if (token.tag_.startswith('NN')) or (token.pos_ == 'ADJ') or (token.pos_ == 'ADV') or (token.pos_ == 'NUM'): #or (token.pos_ == 'VERB'):
                    idxlist.append(idx)

            #candidateslist = [o for o in rep(targettemplatedict['info'][0]['text'], idxlist, 20, dropout=0.2)]

            candidateslist = [o for o in rep(targettemplatedict['info'][0]['text'], idxlist, 20, dropout=0.2)]
            print(candidateslist, flush=True)

            with open(currentpath + '/DonePickle.pkl', 'ab') as f:
                pickle.dump({'entrytemplateidx': entrytemplateidx, 'targettemplatedictidx': targettemplatedictidx, 'candidateslist': candidateslist, 'targettemplatedict': targettemplatedict, 'idxlist': idxlist}, f)

            for candidateidx, candidate in enumerate(candidateslist):
                candidatedatastring = convert_data(candidate, targettemplatedict, idxlist)

                if candidatedatastring == 'ERROR ERROR':
                    break

                elif candidateidx < 1:
                    candidates125list.append([candidate, candidatedatastring])
                    candidates250list.append([candidate, candidatedatastring])
                    candidates500list.append([candidate, candidatedatastring])
                    candidates1000list.append([candidate, candidatedatastring])
                elif candidateidx < 2:
                    candidates250list.append([candidate, candidatedatastring])
                    candidates500list.append([candidate, candidatedatastring])
                    candidates1000list.append([candidate, candidatedatastring])
                elif candidateidx < 5:
                    candidates500list.append([candidate, candidatedatastring])
                    candidates1000list.append([candidate, candidatedatastring])
                elif candidateidx < 10:
                    candidates1000list.append([candidate, candidatedatastring])
                else:
                    break

    rep = None  # NOTE: clear out GPU memory

    candidatesdict = {'125': candidates125list, '250': candidates250list, '500': candidates500list, '1000': candidates1000list}

    for candlist in candidatesdict:
        candidatestrg = [x[0] for x in candidatesdict[candlist]]
        candidatessrc = [x[1] for x in candidatesdict[candlist]]

        alltrgstring = '\n'.join(candidatestrg)
        allsrcstring = '\n'.join(candidatessrc)

        with open(currentpath + '/Predictions/Extended' + candlist + '_trg.txt', 'wb') as f:
            f.write(bytes(alltrgstring, 'UTF-8'))

        with open(currentpath + '/Predictions/Extended' + candlist + '_src.txt', 'wb') as f:
            f.write(bytes(allsrcstring, 'UTF-8'))

def collect_dict():
    currentpath = os.getcwd()

    with open(currentpath + '/Data/AllEntryTemplateInfo.json', 'r') as infile:
        allentrytemplateinfo = json.load(infile)

    fulltrain = []

    for e2efile in allentrytemplateinfo:
        if '\\train\\' in e2efile:
            fulltrain += allentrytemplateinfo[e2efile]

    data_augmentation(fulltrain)



#allentrytemplateinfo = main()
collect_dict()