import numpy as np
from tqdm import tqdm
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from flair.data import Sentence
from flair.models import SequenceTagger

import sys
import os
import pickle
from collections import OrderedDict
import json

from lm import LanguageModel as Model
import utils


def check_sentence(dataset, tokens):
  periods = 0
  triple_dots = 0
  length = len(tokens)
  symbols = 0
  mname = 0

  for token in tokens:
    if token == '.':
      periods += 1
    if token == '...':
      triple_dots += 1
    symbol = re.sub("[A-Za-z0-9]", '', token)
    if len(symbol) > 0 and symbol not in ",!.'":
      symbols += 1

  if dataset == 'rotten':
    return periods == 1 and triple_dots == 0 and length >= 20 and length <= 30 and symbols <= 2 # rotten tomatoes
  else:
    return symbols < 3 and length >= 50 and length <= 90 # yelp


def nuclear_filter(probs):
  indices = np.argsort(probs)[::-1]
  cump = 0

  new_indices = []
  new_probs = []
  for idx in indices:
    if idx < 4:
      continue
    cump += probs[idx]
    if cump > 0.9:
      break
    new_indices.append(idx)
    new_probs.append(probs[idx])
  if len(new_indices) == 0:
    new_indices.append(indices[0])
    new_probs.append(probs[indices[0]])
  new_probs /= np.sum(new_probs)

  return new_probs, new_indices


def sample(probs, indices):
  cumsum = np.cumsum(probs)
  rdm_unif = np.random.rand()
  res = np.searchsorted(cumsum, rdm_unif)
  try:
    return new_indices[res]
  except:
    return indices[0]


def replace_tokens(chunks, probs, probs_indices, replace_rate=0.8):
  idx = 0
  new_chunks = []
  for chunk in chunks:
    new_chunk = []
    for token in chunk:
      p_sub = np.random.rand()
      if p_sub > replace_rate:
        new_chunk.append(token)
      else:
        new_idx = sample(probs[idx], probs_indices[idx])
        new_chunk.append(new_idx)
      idx += 1
    new_chunks.append(new_chunk)

  return new_chunks


def split_to_chunks(tokens, tags, chunk_dict, grammar_set):
  chunks = []
  ctags = []

  cur_chunk = []
  cur_ctag = ""
  for token, tag in zip(tokens, tags):
    spl = tag[1:-1].split('-')
    cidx = spl[0]
    ctag = ''.join(spl[1:])

    if cidx == 'B' or cidx == 'S':
      if len(cur_chunk) != 0:
        cur_chunk = tuple(cur_chunk)
        chunks.append(cur_chunk)
        ctags.append(cur_ctag)
        if cur_ctag not in chunk_dict:
          chunk_dict[cur_ctag] = []
        chunk_dict[cur_ctag].append(cur_chunk)
      cur_chunk = [token]
      cur_ctag = ctag
    else:
      cur_chunk.append(token)
  if len(cur_chunk) != 0:
    cur_chunk = tuple(cur_chunk)
    chunks.append(cur_chunk)
    ctags.append(cur_ctag)
    if cur_ctag not in chunk_dict:
      chunk_dict[cur_ctag] = []
    chunk_dict[cur_ctag].append(cur_chunk)
  grammar_set.append(tuple(ctags))

  return chunks, ctags, chunk_dict, grammar_set


def remove_chunks(chunks, ctags, remove_rate=0.4):
  new_chunks = []
  new_ctags = []

  p_subs = np.random.rand(len(chunks))
  for i, (chunk, ctag) in enumerate(zip(chunks, ctags)):
    p_sub = p_subs[i]
    if p_sub > remove_rate:
      new_chunks.append(chunk)
      new_ctags.append(ctag)

  return new_chunks, new_ctags


def insert_chunks(chunks, ctags, chunk_dict, grammar_set,
                  rands, rand_idx, gidx):
  grammar = grammar_set[gidx]
  gidx += 1
  if gidx == len(grammar_set):
    gidx = 0
  
  sent_chunk_dict = {}
  for ctag, chunk in zip(ctags, chunks):
    if ctag not in sent_chunk_dict:
      sent_chunk_dict[ctag] = []
    if chunk not in sent_chunk_dict[ctag]:
      sent_chunk_dict[ctag].append(chunk)

  new_sentence = []
  found = False
  for ctag in grammar:
    if ctag not in sent_chunk_dict:
      chunk_list = chunk_dict[ctag]
      found = False
    else:
      chunk_list = sent_chunk_dict[ctag]
      found = True
    cidx = int(rands[rand_idx] * len(chunk_list))
    rand_idx += 1
    if rand_idx == 10000:
      rand_idx = 0
    new_sentence += list(chunk_list[cidx])
    if found:
      del chunk_list[cidx]
      if len(sent_chunk_dict[ctag]) == 0:
        del sent_chunk_dict[ctag]

  return new_sentence, chunk_dict, grammar_set, rands, rand_idx, gidx


def chunk_text(file):
  f = open(file, 'r', encoding='utf-8', errors='ignore')
  data = json.load(f)
  f.close()

  tokens_list = []
  tags_list = []

  tagger = SequenceTagger.load('chunk')
  for inst in data:
    reviews = inst['reviews']

    for review in reviews:
      tokens = ["<s>"] + review.split() + ["</s>"]
      tokens_list.append(tokens)

      sentence = Sentence(review)
      tags = [token.annotation_layers['np'][0].value for token in tokens]

      tokens_list.append(tokens)
      tags_list.append(tags)

  return tokens_list, tags_list


def sample(corpus, N):
    shuffle_indices = np.random.permutation(np.arange(len(corpus)))
    return np.array(corpus)[shuffle_indices][:N]


def rouge_1_idf(a, b, idf_dict):
    a = set(a.split()) # summary/query
    b = set(b.split()) # review/key
    c = a.intersection(b)
    weighted_overlap = 0
    for token in c:
        weighted_overlap += idf_dict[token]
    highest_possible = 0
    for token in a:
        highest_possible += idf_dict[token]
    return weighted_overlap / highest_possible


def segment_noise(dataset, summary):
  chunk_dict = {}
  grammar_set = []

  rands = np.random.rand(10000)
  rand_idx = 0
  gidx = 0

  batch_size = 128

  file_dir = 'data/' + dataset + '/'
  model_file = 'model/%s/lm.model' % dataset
  dict_file = 'model/%s/lm.dict.p' % dataset
  train_file = 'data/%s/train.json' % dataset

  tokens_list, tags_list = chunk_text(train_file)
  token_dict = pickle.load(open(dict_file, 'rb'))

  word_size = len(token_dict)
  word_dim = 256
  hidden_dim = 512

  model = Model(word_size, word_dim, hidden_dim)
  model.cuda()
  if os.path.exists(model_file):
    best_point = torch.load(model_file)
    state_dict = best_point['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
      temp = state_dict[k]
      if k.startswith('module.'):
        k = k[7:]
        new_state_dict[k] = temp
    model.load_state_dict(new_state_dict)
  model.eval()

  shuffle_indices = np.random.permutation(np.arange(len(tokens_list)))
  tokens_list = np.array(tokens_list)[shuffle_indices]
  tags_list = np.array(tags_list)[shuffle_indices]

  noised_data = []

  rev_token_dict = {token_dict[token]:token for token in token_dict}
  for _ in range(1):
    for idx in tqdm(range(0, len(tokens_list), batch_size)):
      tokens_batch = tokens_list[idx:idx+batch_size]
      tags_batch = tags_list[idx:idx+batch_size]

      probs_batch = []
      probs_indices_batch = []
      for tokens in tqdm(tokens_batch):
        if not check_sentence([rev_token_dict[token] for token in tokens]):
          continue

        x_batch, x_mask = utils.pad([tokens])
        x_batch = to_tensor(x_batch)
        x_mask = to_tensor(x_mask).float()

        ps_batch = model(x_batch, x_mask, ps_only=True)
        ps_batch = F.softmax(ps_batch, dim=-1)
        ps_batch = list(ps_batch.cpu().detach().numpy())

        probs_sequence = []
        probs_indices_sequence = []
        for ps in ps_batch[0]:
          probs, probs_indices = nuclear_filter(ps)
          probs_sequence.append(probs)
          probs_indices_sequence.append(probs_indices)

        probs_batch.append(probs_sequence)
        probs_indices_batch.append(probs_indices_sequence)
      
      chunk_dict = {}
      grammar_set = []

      chunks_batch = []
      ctags_batch = []
      for tokens, tags in zip(tokens_batch, tags_batch):
        chunks, ctags = split_to_chunks(tokens[1:-1], tags, chunk_dict, grammar_set)
        chunks_batch.append(chunks)
        ctags_batch.append(ctags)

      for chunk in chunk_dict:
        chunk_dict[chunk] = list(set(chunk_dict[chunk]))
      grammar_set = list(set(grammar_set))
      np.random.shuffle(grammar_set)

      ps_idx = 0
      for j, (tokens, chunks, ctags) in enumerate(tqdm(zip(tokens_batch, chunks_batch, ctags_batch), total=len(chunks_batch))):
        if not check_sentence([rev_token_dict[token] for token in tokens]):
          continue

        lm_chunk_inputs = []

        probs = probs_batch[ps_idx]
        probs_indices = probs_indices_batch[ps_idx]
        ps_idx += 1
        if dataset == 'rotten':
          N = 20
        else:
          N = 8
        for _ in tqdm(range(N)):
          try:
            new_chunks = replace_tokens(chunks, probs, probs_indices)

            new_chunks, new_ctags = remove_chunks(new_chunks, ctags)
            lm_chunk_input = insert_chunks(new_chunks, new_ctags, chunk_dict, grammar_set,
                                           rands, rand_idx, gidx)
            lm_chunk_input = ' '.join([rev_token_dict[token] for token in lm_chunk_input])
            lm_chunk_inputs.append(lm_chunk_input)
          except:
            pass

        inst = {}
        inst['summary'] = ' '.join([rev_token_dict[token] for token in tokens[1:-1]])
        inst['segment_reviews'] = lm_chunk_inputs

        noised_data.append(inst)

  return noised_data


def document_noise(dataset, noised_data):
  train_file = 'data/%s/train.json' % dataset
  if dataset == 'rotten':
    N = 20
  else:
    N = 8

  f = open(train_file, 'r', encoding='utf-8', errors='ignore')
  data = json.load(f)
  f.close()

  corpus = []
  for inst in data:
    for review in inst['reviews']:
      corpus.append(review)

  idf_dict = {}
  for text in tqdm(corpus):
    tokens = set(text.split())
    for token in tokens:
      if token not in idf_dict:
        idf_dict[token] = 0
      idf_dict[token] += 1

  for token in tqdm(idf_dict):
    idf_dict[token] = -np.log(idf_dict[token] / len(corpus))

  inp_file = file_dir + '/rotten.train.wasa.json'
  out_file = file_dir + '/rotten.train.wasapee.json'

  data_index = {}
  for idx, inst in enumerate(noised_data):
    if inst['summary'] not in data_index:
      data_index[inst['summary']] = []
    data_index[inst['summary']].append(idx)

  count = 0
  for idx, text in enumerate(tqdm(corpus)):
    if text not in data_index:
      continue
    inst = noised_data[data_index[text].pop()]
    if len(data_index[text]) == 0:
      del data_index[text]

    a = inst['summary']
    sample_corpus = corpus[max(0, idx-128) : idx] + corpus[idx+1 : min(len(corpus), idx+128)]
    sorted_corpus = sorted(sample_corpus, key=lambda b: -rouge_1_idf(a, b, idf_dict))
    inst['document_reviews'] = sorted_corpus[:N]

    count += 1

  for inst in tqdm(noised_data):
    if 'document_reviews' in inst:
      continue
      
    a = inst['summary']
    sample_corpus = sample(corpus, N=10000)
    sorted_corpus = sorted(sample_corpus, key=lambda b: -rouge_1_idf(a, b, idf_dict))
    inst['document_reviews'] = sorted_corpus[:N]