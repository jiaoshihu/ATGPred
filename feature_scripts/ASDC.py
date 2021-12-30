#!/usr/bin/env python
#_*_coding:utf-8_*_


def get_ASDC(fastas):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    aaPairs = []
    for aa1 in AA:
        for aa2 in AA:
            aaPairs.append(aa1 + aa2)
    for i in fastas:
        sequence = re.sub('-', '', i)
        code = []
        sum = 0
        pair_dict = {}
        for pair in aaPairs:
            pair_dict[pair] = 0
        for j in range(len(sequence)):
            for k in range(j + 1, len(sequence)):
                if sequence[j] in AA and sequence[k] in AA:
                    pair_dict[sequence[j] + sequence[k]] += 1
                    sum += 1
        for pair in aaPairs:
            code.append(pair_dict[pair] / sum)
        encodings.append(code)
    return encodings



























import re

def DPC(fastas, **kw):
    AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    header = ['#', 'label'] + diPeptides
    encodings.append(header)

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    for i in fastas:
        name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
        code = [name, label]
        tmpCode = [0] * 400
        for j in range(len(sequence) - 2 + 1):
            tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] +1
        if sum(tmpCode) != 0:
            tmpCode = [i/sum(tmpCode) for i in tmpCode]
        code = code + tmpCode
        encodings.append(code)
    return encodings