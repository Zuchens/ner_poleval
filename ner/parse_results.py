import json
import sys, json, getopt
from dateutil import parser
from tqdm import tqdm
from ner.treebank_span import TreebankSpanTokenizer


def makeAnnsFormat(inputDoc, cols, htype):
    z_anns = []
    for ben in inputDoc.split('\n'):
        pcs = ben.split('\t')
        try:
            if len(pcs)==cols:
                cat, ofrom, oto = pcs[-2].split(' ')
                z_anns.append( {"from": ofrom, "to" :oto, "type":  cat,  "text":pcs[-1]})
        except ValueError:
            # handling fragmented entity, two strategies:
            if htype=='merge':
                # take start and end, use as a single big entity
                cat, ofrom, ignored, oto = pcs[-2].split(' ')
                z_anns.append(  {"from": ofrom, "to" :oto, "type":  cat,  "text":pcs[-1]} )
            if htype=='split':
                # split into two entities
                catAndOffsets1, offsets2 = pcs[-2].split(';')
                cat, ofrom, oto = catAndOffsets1.split(' ')
                z_anns.append(  {"from": ofrom, "to" :oto, "type":  cat,  "text":pcs[-1]} )
                ofrom, oto = offsets2.split(' ')
                z_anns.append(  {"from": ofrom, "to" :oto, "type":  cat,  "text":pcs[-1]} )
    return z_anns


def compute(goldfile, htype="split"):
    idsToAnnsGold = {"texts" : []}
    with open(goldfile) as json_data:
        goldjson = json.load(json_data)

    for nr in range(len(goldjson['questions'])):
        idGold = '/'.join(goldjson['questions'][nr]['input']['fname'].split('/')[4:])
        if len(goldjson['questions'][nr]['answers']) > 1:
            maximum = parser.parse('1900-01-02T14:22:41.439308+00:00')
            index = 0
            for i, value in enumerate(goldjson['questions'][nr]['answers']):
                value = parser.parse(goldjson['questions'][nr]['answers'][i]['created'])
                if value > maximum:
                    maximum = value
                    index = i
            ans = goldjson['questions'][nr]['answers'][index]['data']['brat']
        else:
            ans = goldjson['questions'][nr]['answers'][0]['data']['brat']

        text = goldjson['questions'][nr]['input']['fileContent']
        tokenizer = TreebankSpanTokenizer()
        text_tokenized = tokenizer.tokenize(text)
        span_tokenized = [x for x in tokenizer.span_tokenize(text)]
        ovtp = makeAnnsFormat(ans, 3, htype)
        offsets2Entities = {}
        entities = []
        idx = []
        for x in span_tokenized:
            data = []
            entities.append(data)
            offsets2Entities[str(x[0])] = []
            idx.append(x[0])
        for val in tqdm(ovtp):
            start = val["from"]
            end = val["to"]
            types = val["type"].split("_")
            val["type"] = types[0]
            if(len(types)>1):
                val["subtype"] = types[1]
            i =0
            while i < len(idx):
                if idx[i] == int(start):
                    while i < len(idx) and idx[i] < int(end):
                        entities[i].append(val)
                        offsets2Entities[str(idx[i])].append(val)
                        i += 1
                    break
                i += 1
        idsToAnnsGold["texts"].append({"tokens":text_tokenized, "entities": entities , "text" : text})
    with open("results.json", "w") as f:
        json.dump(idsToAnnsGold, f, indent = 2, ensure_ascii=False)

compute("POLEVAL-NER_GOLD.json")
