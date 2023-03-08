import pickle
import os
import regex as re
from bleurt import score
from sacrebleu.metrics import BLEU
from bert_score import score as bertscore
from statistics import mean

currentpath = os.getcwd()

#domainlist = ['Incidents', 'Sports', 'Stocks', 'Weather']
#domainlist = ['Airport', 'Astronaut', 'Building', 'City', 'ComicsCharacter', 'Food', 'Monument', 'SportsTeam', 'University', 'WrittenWork']
domainlist = ['']
for domain in domainlist:
    previousdonelist = []
    with open(currentpath + '/DonePickle' + domain + '.pkl', 'rb') as fr:
        try:
            while True:
                previousdonelist.append(pickle.load(fr))
        except EOFError:
            pass

    candidates125list = []
    candidates250list = []
    candidates500list = []
    candidates1000list = []
    refs125list = []
    refs250list = []
    refs500list = []
    refs1000list = []

    for prevdoneentry in previousdonelist:
        candidateslist = prevdoneentry['candidateslist']
        #reference = prevdoneentry['targettemplatedict']['text']
        reference = prevdoneentry['targettemplatedict']['info'][0]['text']
        for candidateidx, candidate in enumerate(candidateslist):
            candidatetxt = re.sub(r'\n|\r', ' ', candidate)
            #candidatetxt = wordtokenizer(candidatetxt)
            #candidatetxt = md.detokenize(candidatetxt)
            refstxt = re.sub(r'\n|\r', ' ', reference)
            #refstxt = wordtokenizer(refstxt)
            #refstxt = md.detokenize(refstxt)

            if candidateidx < 1:
                candidates125list.append(candidatetxt)
                candidates250list.append(candidatetxt)
                candidates500list.append(candidatetxt)
                candidates1000list.append(candidatetxt)
                refs125list.append(refstxt)
                refs250list.append(refstxt)
                refs500list.append(refstxt)
                refs1000list.append(refstxt)
            elif candidateidx < 2:
                candidates250list.append(candidatetxt)
                candidates500list.append(candidatetxt)
                candidates1000list.append(candidatetxt)
                refs250list.append(refstxt)
                refs500list.append(refstxt)
                refs1000list.append(refstxt)
            elif candidateidx < 5:
                candidates500list.append(candidatetxt)
                candidates1000list.append(candidatetxt)
                refs500list.append(refstxt)
                refs1000list.append(refstxt)
            elif candidateidx < 10:
                candidates1000list.append(candidatetxt)
                refs1000list.append(refstxt)

    checkpoint = currentpath + "/BLEURT-20"
    scorer = score.BleurtScorer(checkpoint)

    lengthlist = ['1000']
    for trainlength in lengthlist:
        if trainlength == '125':
            scores = scorer.score(references=refs125list, candidates=candidates125list)
            assert type(scores) == list and len(scores) == len(refs125list)
            print(domain + ' ' + trainlength, flush=True)
            print('BLEURT: ' + str(mean(scores)), flush=True)
            bleu = BLEU()
            bleuscore = bleu.corpus_score(candidates125list, refs125list)
            print(bleuscore, flush=True)
            bertp, bertr, bertf1 = bertscore(candidates125list, refs125list, lang='en', verbose=True,
                                             rescale_with_baseline=True)
            print(f"BERT Precision score: {bertp.mean():.3f}", flush=True)
            print(f"BERT Recall score: {bertr.mean():.3f}", flush=True)
            print(f"BERT F1 score: {bertf1.mean():.3f}", flush=True)
        elif trainlength == '250':
            scores = scorer.score(references=refs250list, candidates=candidates250list)
            assert type(scores) == list and len(scores) == len(refs250list)
            print(domain + ' ' + trainlength, flush=True)
            print('BLEURT: ' + str(mean(scores)), flush=True)
            bleu = BLEU()
            bleuscore = bleu.corpus_score(candidates250list, refs250list)
            print(bleuscore, flush=True)
            bertp, bertr, bertf1 = bertscore(candidates250list, refs250list, lang='en', verbose=True,
                                             rescale_with_baseline=True)
            print(f"BERT Precision score: {bertp.mean():.3f}", flush=True)
            print(f"BERT Recall score: {bertr.mean():.3f}", flush=True)
            print(f"BERT F1 score: {bertf1.mean():.3f}", flush=True)
        elif trainlength == '500':
            scores = scorer.score(references=refs500list, candidates=candidates500list)
            assert type(scores) == list and len(scores) == len(refs500list)
            print(domain + ' ' + trainlength, flush=True)
            print('BLEURT: ' + str(mean(scores)), flush=True)
            bleu = BLEU()
            bleuscore = bleu.corpus_score(candidates500list, refs500list)
            print(bleuscore, flush=True)
            bertp, bertr, bertf1 = bertscore(candidates500list, refs500list, lang='en', verbose=True,
                                             rescale_with_baseline=True)
            print(f"BERT Precision score: {bertp.mean():.3f}", flush=True)
            print(f"BERT Recall score: {bertr.mean():.3f}", flush=True)
            print(f"BERT F1 score: {bertf1.mean():.3f}", flush=True)
        elif trainlength == '1000':
            scores = scorer.score(references=refs1000list, candidates=candidates1000list)
            assert type(scores) == list and len(scores) == len(refs1000list)
            print(domain + ' ' + trainlength, flush=True)
            print('BLEURT: ' + str(mean(scores)), flush=True)
            refs1000listbleu = [[x] for x in refs1000list]
            bleu = BLEU()
            bleuscore = bleu.corpus_score(candidates1000list, refs1000listbleu)
            print(bleuscore, flush=True)
            bertp, bertr, bertf1 = bertscore(candidates1000list, refs1000list, lang='en', verbose=True,
                                             rescale_with_baseline=True)
            print(f"BERT Precision score: {bertp.mean():.5f}", flush=True)
            print(f"BERT Recall score: {bertr.mean():.5f}", flush=True)
            print(f"BERT F1 score: {bertf1.mean():.5f}", flush=True)


