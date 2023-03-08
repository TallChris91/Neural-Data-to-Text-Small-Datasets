from typing import List, Tuple, FrozenSet
from argparse import ArgumentParser, Namespace
import random
import sys
import json
import os
import re

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def preprocess_ssgp(fact_set: FrozenSet[Tuple[str, str]]) -> FrozenSet[Tuple[str, str]]:
    kept_facts = []
    for fact in fact_set:
        if not (fact[0].endswith("'") or fact[1].endswith("'")):
            kept_facts.append(fact)
    return frozenset(kept_facts)


def lemmatize_relations(fact_set: FrozenSet[Tuple[str, str]])\
        -> FrozenSet[Tuple[str, str]]:
    new_facts = []
    for fact in fact_set:
        new_facts.append(
            (lemmatizer.lemmatize(fact[1], pos='v'), fact[2])
        )
    return frozenset(new_facts)


def generate_triple(elements: List[str], lower: bool = True) -> Tuple[str, str]: #SET LOWER BACK TO TRUE FOR P/R/F1!
    if elements:
        prd = elements.pop(0)
    else:
        prd = ''
    if elements:
        obj = elements.pop(0)
    else:
        obj = ''

    if lower:
        return prd.lower(), obj.lower()
    else:
        return prd, obj


def factseq2set(seq: str, eof=" @EOF@ ", sep=" @SEP@ ",
                lower=True) -> FrozenSet[Tuple[str, str]]:
    fact_strings: List[str] = seq.split(eof)
    facts: List[Tuple[str, str]] = [
        generate_triple(fact_string.split(sep))
        for fact_string in fact_strings
    ]
    return frozenset(facts)


def main(args: Namespace):
    num_correct_predicted = 0
    num_gt_facts = 0
    num_predicted_facts = 0

    with open(args.hypo_file) as hf, open(args.ref_file) as rf:
        data = list(zip(hf.readlines(), rf.readlines()))

    if args.val100:
        rand = random.Random(0)
        data = rand.sample(data, 100)

    allgeneratedrefs = []
    allgeneratedhypos = []

    for hline, rline in data:
        hypo_facts = factseq2set(hline.strip())
        refs = [
            factseq2set(ref_seq) for ref_seq in rline.strip().split('*#')
        ]
        num_gt_facts += max(len(r) for r in refs)

        if args.ssgp:
            hypo_facts = preprocess_ssgp(hypo_facts)
            refs = [lemmatize_relations(r) for r in refs]

        #hypo_facts = [x for x in hypo_facts if x != '']
        #hypo_facts = [x for x in hypo_facts if not('@' in x)]

        new_hypo_facts = []

        for triple in hypo_facts:
            skiptriple = 'n'
            for element in triple:
                if element == '':
                    skiptriple = 'y'
                elif '@' in element:
                    skiptriple = 'y'

            if skiptriple == 'n':
                new_hypo_facts.append(triple)

        new_hypo_facts = frozenset(new_hypo_facts)
        hypo_facts = new_hypo_facts.copy()

        num_predicted_facts += len(hypo_facts)

        allgeneratedhypos.append(list(hypo_facts))
        newrefs = [list(x) for x in refs]
        allgeneratedrefs.append(newrefs)

        for hfact in hypo_facts:
            if any(hfact in r for r in refs):
                num_correct_predicted += 1

    precision = float(num_correct_predicted) / \
        float(num_predicted_facts + 1e-13)
    recall = float(num_correct_predicted) / float(num_gt_facts + 1e-13)
    f1 = 2. * ((precision * recall) / (precision + recall + 1e-13))

    print(
        "P: {:.1f} / R: {:.1f} / F1: {:.1f}".format(
            precision * 100, recall * 100, f1 * 100
        ), file=sys.stderr
    )

    scorestring = 'Precision: ' + (str(precision * 100)) + '\n'
    scorestring += 'Recall: ' + (str(recall * 100)) + '\n'
    scorestring += 'F1: ' + (str(f1 * 100))

    newfile = os.path.splitext(args.ref_file)[0]
    newfile = re.sub(r'\_refs$', '', newfile)

    with open(newfile + '_score.txt', 'w') as hf:
        hf.write(scorestring)

    with open(os.path.splitext(args.hypo_file)[0]+'_modified.json', 'w') as outfile:
        json.dump(allgeneratedhypos, outfile, indent=4, separators=(',', ': '))

    with open(os.path.splitext(args.ref_file)[0]+'_modified.json', 'w') as outfile:
        json.dump(allgeneratedrefs, outfile, indent=4, separators=(',', ': '))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('hypo_file')
    parser.add_argument('ref_file')
    parser.add_argument('--val100', action='store_true')
    parser.add_argument('--ssgp', action='store_true')
    args = parser.parse_args()
    main(args)
