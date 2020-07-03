import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class DenoiseSum(nn.Module):

  def __init__(self, word_size, word_dim, hidden_dim, disc_size):
    super(DenoiseSum, self).__init__()
    self.word_size = word_size
    self.word_dim = word_dim
    self.hidden_dim = hidden_dim
    self.fuse = fuse
    self.disc_type = disc_type

    # EMBEDDING
    self.word_embs = nn.Embedding(num_embeddings=word_size,
                                  embedding_dim=word_dim,
                                  padding_idx=0)

    # ENCODER
    self.fw_encoder = nn.LSTMCell(word_dim, hidden_dim//2)
    self.bw_encoder = nn.LSTMCell(word_dim, hidden_dim//2)

    # DENOISING
    self.dns_att_key = nn.Linear(hidden_dim, hidden_dim)
    self.dns_att_query = nn.Linear(hidden_dim, hidden_dim)
    self.dns_pnt_key = nn.Linear(hidden_dim, hidden_dim)
    self.dns_pnt_query = nn.Linear(hidden_dim, hidden_dim)

    # FUSION
    self.fse_att_key = nn.Linear(hidden_dim, hidden_dim)
    self.fse_pnt_key = nn.Linear(hidden_dim, hidden_dim)
    self.fse_att_transform = nn.Linear(hidden_dim, hidden_dim//2)
    self.fse_pnt_transform = nn.Linear(hidden_dim, hidden_dim//2)

    # DISCRIMINATOR
    self.dsc_intermediate = nn.Linear(hidden_dim, hidden_dim)
    self.dsc_classifier = nn.Linear(hidden_dim, disc_size)

    # SEGMENT DECODER
    self.att_decoder = nn.LSTMCell(word_dim, hidden_dim//2)

    self.att_key = nn.Linear(hidden_dim, hidden_dim//2)
    self.att_query = nn.Linear(hidden_dim//2, hidden_dim//2)
    self.att_weight = nn.Linear(hidden_dim//2, 1)

    self.att_ctx_classifier = nn.Linear(hidden_dim, word_size)
    self.att_hid_classifier = nn.Linear(hidden_dim//2, word_size)

    self.att_coverage = nn.Linear(1, hidden_dim//2)

    # DOCUMENT DECODER
    self.pnt_decoder = nn.LSTMCell(word_dim, hidden_dim//2)

    self.pnt_key = nn.Linear(hidden_dim, hidden_dim//2)
    self.pnt_query = nn.Linear(hidden_dim//2, hidden_dim//2)
    self.pnt_weight = nn.Linear(hidden_dim//2, 1)

    self.pnt_ctx_classifier = nn.Linear(hidden_dim, word_size)
    self.pnt_hid_classifier = nn.Linear(hidden_dim//2, word_size)
    self.pnt_gate = nn.Linear(hidden_dim*3//2, 1)

    self.pnt_coverage = nn.Linear(1, hidden_dim//2)

    self.final_gate = nn.Linear(hidden_dim*3//2, 1)

    # OTHERS
    self.dropout = nn.Dropout(0.1)


  def load_partial_state_dict(self, state_dict):
    own_state = self.state_dict()
    for name, param in state_dict.items():
      if name.startswith("module."):
        own_name = name[7:]
      else:
        own_name = name
      own_state[own_name].copy_(state_dict[name].data)


  def encoder_lm(self, input, mask):
    batch_size, inp_len = input.size()

    # EMBED
    xs = self.word_embs(input)
    xs = self.dropout(xs)

    # ENCODE
    ## FORWARD
    ht = xs.new_zeros(batch_size, self.hidden_dim//2)
    ct = xs.new_zeros(batch_size, self.hidden_dim//2)
    fw_hs = []
    for i in range(inp_len):
      xt = xs[:,i]
      ht_prev = ht*1
      ht, ct = self.fw_encoder(xt, (ht, ct))
      fw_hs = fw_hs + [ht]
    fw_hs = torch.stack(fw_hs, dim=1)

    ## BACKWARD
    ht = xs.new_zeros(batch_size, self.hidden_dim//2)
    ct = xs.new_zeros(batch_size, self.hidden_dim//2)
    bw_hs = []
    for i in reversed(range(inp_len)):
      xt = xs[:,i]
      ht_prev = ht*1
      ht, ct = self.bw_encoder(xt, (ht, ct))
      bw_hs = [ht] + bw_hs
    bw_hs = torch.stack(bw_hs, dim=1)

    hs = torch.cat([fw_hs[:,:-2], bw_hs[:,2:]], dim=-1)
    hs = self.dropout(hs)

    # CLASSIFY
    ps = self.att_ctx_classifier(hs) + self.pnt_ctx_classifier(hs)

    inp_len -= 2
    ps = ps.view(batch_size*inp_len, -1)
    ys = input[:,1:-1].contiguous().view(batch_size*inp_len)
    mask = mask[:,1:-1].contiguous().view(batch_size*inp_len)

    loss = F.cross_entropy(ps, ys, reduction='none')
    loss *= mask
    loss = loss.mean()

    return loss


  def decoder_lm(self, input, mask):
    batch_size, inp_len = input.size()

    # EMBED
    xs = self.word_embs(input)
    xs = self.dropout(xs)

    # DECODE
    st_att = xs.new_zeros(batch_size, self.hidden_dim//2)
    st_pnt = xs.new_zeros(batch_size, self.hidden_dim//2)
    ct_att = xs.new_zeros(batch_size, self.hidden_dim//2)
    ct_pnt = xs.new_zeros(batch_size, self.hidden_dim//2)
    ps = []
    for i in range(inp_len-1):
      xt = xs[:,i]

      st_att, ct_att = self.att_decoder(xt, (st_att, ct_att))
      st_pnt, ct_pnt = self.pnt_decoder(xt, (st_pnt, ct_pnt))

      ## CLASSIFY
      pt = self.att_hid_classifier(st_att) + self.pnt_hid_classifier(st_pnt)
      ps.append(pt)
    ps = torch.stack(ps, dim=1)

    inp_len -= 1
    ps = ps.view(batch_size*inp_len, -1)
    ys = input[:,1:].contiguous().view(batch_size*inp_len)
    mask = mask[:,1:].contiguous().view(batch_size*inp_len)

    loss = F.cross_entropy(ps, ys, reduction='none')
    loss *= mask
    loss = loss.mean()

    return loss


  def forward(self, input, 
              doc_att_mask, doc_pnt_mask,
              tok_att_mask, tok_pnt_mask,
              output, output_mask,
              category,
              coverage_rate=0):
    batch_size, doc_len, tok_len = input.size()
    _, out_len = output.size()

    enc_mask = tok_att_mask.view(batch_size*doc_len, tok_len, 1)
    enc_mask += tok_pnt_mask.view(batch_size*doc_len, tok_len, 1)
    enc_mask = torch.min(enc_mask.new([1]), enc_mask)

    # EMBED
    xs = self.word_embs(input)
    xs = self.dropout(xs)

    # ENCODE
    xs = xs.view(batch_size*doc_len, tok_len, -1)

    ## FORWARD
    ht = xs.new_zeros(batch_size*doc_len, self.hidden_dim//2)
    ct = xs.new_zeros(batch_size*doc_len, self.hidden_dim//2)
    fw_hs = []
    for i in range(tok_len):
        xt = xs[:,i]
        ht_prev = ht*1
        ht, ct = self.fw_encoder(xt, (ht, ct))
        mask = enc_mask[:,i]
        ht = ht*mask + ht_prev*(1-mask)
        fw_hs = fw_hs + [ht]
    fw_hs = torch.stack(fw_hs, dim=1)

    ## BACKWARD
    ht = xs.new_zeros(batch_size*doc_len, self.hidden_dim//2)
    ct = xs.new_zeros(batch_size*doc_len, self.hidden_dim//2)
    bw_hs = []
    for i in reversed(range(tok_len)):
        xt = xs[:,i]
        ht_prev = ht*1
        ht, ct = self.bw_encoder(xt, (ht, ct))
        mask = enc_mask[:,i]
        ht = ht*mask + ht_prev*(1-mask)
        bw_hs = [ht] + bw_hs
    bw_hs = torch.stack(bw_hs, dim=1)

    ## COMBINE
    ht = torch.cat([fw_hs[:,-1], bw_hs[:,0]], dim=-1)
    hs = torch.cat([fw_hs, bw_hs], dim=-1)

    ht = ht.view(batch_size, doc_len, -1)
    hs = hs.view(batch_size, doc_len*tok_len, -1)
    kt_att = self.att_key(hs)
    kt_pnt = self.pnt_key(hs)

    # FUSE
    doc_att_mask = doc_att_mask.unsqueeze(-1)
    doc_pnt_mask = doc_pnt_mask.unsqueeze(-1)

    ## DENOISE and FUSE for SEGMENT
    ht_att_mean = ht.sum(dim=1, keepdim=True) / doc_att_mask.sum(dim=1, keepdim=True)
    ht_att_gate = self.dns_att_key(ht) + self.dns_att_query(ht_att_mean)
    ht_att = ht + ht_att_gate.tanh()
    st_att_gate = F.softmax(self.fse_att_key(ht_att), dim=1) * doc_att_mask
    st_att_gate = st_att_gate / st_att_gate.sum(dim=1, keepdim=True)
    st_att = (ht_att*st_att_gate).sum(dim=1)

    ## DENOISE and FUSE for DOCUMENT
    ht_pnt_mean = ht.sum(dim=1, keepdim=True) / doc_pnt_mask.sum(dim=1, keepdim=True)
    ht_pnt_gate = self.dns_pnt_key(ht) + self.dns_pnt_query(ht_pnt_mean)
    ht_pnt = ht + ht_pnt_gate.tanh()        
    st_pnt_gate = F.softmax(self.fse_pnt_key(ht_pnt), dim=1) * doc_pnt_mask
    st_pnt_gate = st_pnt_gate / st_pnt_gate.sum(dim=1, keepdim=True)
    st_pnt = (ht_pnt*st_pnt_gate).sum(dim=1)

    st_att = self.fse_att_transform(st_att).tanh()
    st_pnt = self.fse_pnt_transform(st_pnt).tanh()

    # DISCRIMINATE
    if category is not None:
      st = torch.cat([st_pnt, st_pnt], dim=-1)
      z = self.dsc_intermediate(st).tanh()
      z = self.dropout(z)
      z = self.dsc_classifier(z)
      z = F.softmax(z, dim=-1)
      disc_loss = torch.sum(-category*torch.log(z), dim=-1).mean()

    # DECODE
    tok_att_mask = tok_att_mask.view(batch_size, doc_len*tok_len, 1)
    tok_pnt_mask = tok_pnt_mask.view(batch_size, doc_len*tok_len, 1)

    ys = self.word_embs(output)
    ys = self.dropout(ys)

    ct_att = ys.new_zeros(batch_size, self.hidden_dim//2)
    ct_pnt = ys.new_zeros(batch_size, self.hidden_dim//2)
    cov_att = ys.new_zeros(batch_size, doc_len*tok_len, 1)
    cov_pnt = ys.new_zeros(batch_size, doc_len*tok_len, 1)
    cov_loss = []
    ps = []
    unlikely_loss = []
    gammas = []
    deltas = []
    for i in range(out_len-1):
      yt = ys[:,i]

      ## DECODE for SEGMENT
      st_att, ct_att = self.att_decoder(yt, (st_att, ct_att))

      qt = self.att_query(st_att).unsqueeze(1)
      if coverage_rate != 0:
          qt = qt + self.att_coverage(cov_att)
      at = self.att_weight(torch.tanh(kt_att + qt))
      at = F.softmax(at, dim=1) * tok_att_mask
      at = at / at.sum(dim=1, keepdim=True)
      vt = torch.sum(at*hs, dim=1)

      cov_loss.append(torch.min(cov_att, at).sum())
      cov_att = cov_att + at

      p_wasa = self.att_ctx_classifier(vt) + self.att_hid_classifier(st_att)
      p_wasa = F.softmax(p_wasa, dim=-1)

      delta = self.final_gate(torch.cat([st_att, vt], dim=-1)).sigmoid()
      deltas.append(delta)

      ## DECODE for DOCUMENT
      st_pnt, ct_pnt = self.pnt_decoder(yt, (st_pnt, ct_pnt))

      qt = self.pnt_query(st_pnt).unsqueeze(1)
      if coverage_rate != 0:
          qt = qt + self.pnt_coverage(cov_pnt)
      at = self.pnt_weight(torch.tanh(kt_pnt + qt))
      at = F.softmax(at, dim=1) * tok_pnt_mask
      at = at / at.sum(dim=1, keepdim=True)
      vt = torch.sum(at*hs, dim=1)

      cov_loss.append(torch.min(cov_pnt, at).sum())
      cov_pnt = cov_pnt + at

      p_g = self.pnt_ctx_classifier(vt) + self.pnt_hid_classifier(st_pnt)
      p_g = F.softmax(p_g, dim=-1)

      p_c = torch.zeros(batch_size, self.word_size).cuda()
      w_idx = input.view(-1)
      b_idx = torch.arange(0, batch_size).unsqueeze(-1).expand(-1, doc_len*tok_len).contiguous().view(-1)
      s_idx = torch.arange(0, doc_len*tok_len).unsqueeze(-1).expand(-1, batch_size).transpose(0, 1).contiguous().view(-1)
      p_c[b_idx,w_idx] += at.squeeze(-1)[b_idx,s_idx]

      gamma = self.pnt_gate(torch.cat([st_pnt, vt], dim=-1)).sigmoid()
      gammas.append(gamma)
      p_pee = gamma * p_g + (1-gamma) * p_c

      pt = p_pee
      ps.append(pt)

    ps = torch.stack(ps, dim=1)
    gammas = torch.stack(gammas).mean()
    deltas = torch.stack(deltas).mean()

    log_ps = torch.log(ps)
    loss = F.nll_loss(log_ps.view(-1, self.word_size),
                      output[:,1:].contiguous().view(-1),
                      reduction='none')
    loss = loss * output_mask[:,1:].contiguous().view(-1)
    loss = loss.view(batch_size, -1).sum()
    loss = loss / output_mask.sum()

    if category is not None:
      loss += disc_loss

    if coverage_rate != 0:
      cov_loss = torch.stack(cov_loss).mean() * 2
      loss += coverage_rate * cov_loss

    return ps, loss, gammas, deltas


  def beam_search(self, input, 
                  doc_att_mask, doc_pnt_mask,
                  tok_att_mask, tok_pnt_mask,
                  sos, eos, k=5, max_len=100,
                  coverage_rate=0):
    batch_size, doc_len, tok_len = input.size()

    enc_mask = tok_att_mask.view(batch_size*doc_len, tok_len, 1)
    enc_mask += tok_pnt_mask.view(batch_size*doc_len, tok_len, 1)
    enc_mask = torch.min(enc_mask.new([1]), enc_mask)

    # EMBED
    xs = self.word_embs(input)
    xs = self.dropout(xs)

    # ENCODE
    xs = xs.view(batch_size*doc_len, tok_len, -1)

    ## FORWARD
    ht = xs.new_zeros(batch_size*doc_len, self.hidden_dim//2)
    ct = xs.new_zeros(batch_size*doc_len, self.hidden_dim//2)
    fw_hs = []
    for i in range(tok_len):
      xt = xs[:,i]
      ht_prev = ht*1
      ht, ct = self.fw_encoder(xt, (ht, ct))
      mask = enc_mask[:,i]
      ht = ht*mask + ht_prev*(1-mask)
      fw_hs = fw_hs + [ht]
    fw_hs = torch.stack(fw_hs, dim=1)

    ## BACKWARD
    ht = xs.new_zeros(batch_size*doc_len, self.hidden_dim//2)
    ct = xs.new_zeros(batch_size*doc_len, self.hidden_dim//2)
    bw_hs = []
    for i in reversed(range(tok_len)):
      xt = xs[:,i]
      ht_prev = ht*1
      ht, ct = self.bw_encoder(xt, (ht, ct))
      mask = enc_mask[:,i]
      ht = ht*mask + ht_prev*(1-mask)
      bw_hs = [ht] + bw_hs
    bw_hs = torch.stack(bw_hs, dim=1)

    ## COMBINE
    ht = torch.cat([fw_hs[:,-1], bw_hs[:,0]], dim=-1)
    hs = torch.cat([fw_hs, bw_hs], dim=-1)

    ht = ht.view(batch_size, doc_len, -1)
    hs = hs.view(batch_size, doc_len*tok_len, -1)
    kt_att = self.att_key(hs)
    kt_pnt = self.pnt_key(hs)

    # FUSE
    doc_att_mask = doc_att_mask.unsqueeze(-1)
    doc_pnt_mask = doc_pnt_mask.unsqueeze(-1)

    ## DENOISE and FUSE for SEGMENT
    ht_att_mean = ht.sum(dim=1, keepdim=True) / doc_att_mask.sum(dim=1, keepdim=True)
    ht_att_gate = self.dns_att_key(ht) + self.dns_att_query(ht_att_mean)
    ht_att = ht + ht_att_gate.tanh()
    st_att_gate = F.softmax(self.fse_att_key(ht_att), dim=1) * doc_att_mask
    st_att_gate = st_att_gate / st_att_gate.sum(dim=1, keepdim=True)
    st_att = (ht_att*st_att_gate).sum(dim=1)

    ## DENOISE and FUSE for DOCUMENT
    ht_pnt_mean = ht.sum(dim=1, keepdim=True) / doc_pnt_mask.sum(dim=1, keepdim=True)
    ht_pnt_gate = self.dns_pnt_key(ht) + self.dns_pnt_query(ht_pnt_mean)
    ht_pnt = ht + ht_pnt_gate.tanh()
    st_pnt_gate = F.softmax(self.fse_pnt_key(ht_pnt), dim=1) * doc_pnt_mask
    st_pnt_gate = st_pnt_gate / st_pnt_gate.sum(dim=1, keepdim=True)
    st_pnt = (ht_pnt*st_pnt_gate).sum(dim=1)

    st_att = self.fse_att_transform(st_att).tanh()
    st_pnt = self.fse_pnt_transform(st_pnt).tanh()

    # DECODE
    tok_att_mask = tok_att_mask.view(batch_size, doc_len*tok_len, 1)
    tok_pnt_mask = tok_pnt_mask.view(batch_size, doc_len*tok_len, 1)

    ct_att = xs.new_zeros(batch_size, self.hidden_dim//2)
    ct_pnt = xs.new_zeros(batch_size, self.hidden_dim//2)
    cov_att = xs.new_zeros(batch_size, doc_len*tok_len, 1)
    cov_pnt = xs.new_zeros(batch_size, doc_len*tok_len, 1)

    beam = []
    beam.append({
      'input': torch.Tensor([sos]).cuda(),
      'states': (st_att, ct_att, st_pnt, ct_pnt),
      'coverage': (cov_att, cov_pnt),
      'sequence': [],
      'prob': 0,
      'prob_norm': 0,
      'trigrams': []
    })
    finished = []
    while len(beam) != 0:
      new_beam = []
      for instance in beam:
        yt = instance['input']
        st_att, ct_att, st_pnt, ct_pnt = instance['states']
        cov_att, cov_pnt = instance['coverage']
        sequence = instance['sequence']
        prob = instance['prob']
        prob_norm = instance['prob_norm']
        trigrams = instance['trigrams']

        if len(sequence) == max_len:
          finished.append(instance)
          continue
        if yt == eos:
          finished.append(instance)
          continue

        yt = self.word_embs(yt)

        ## DECODE for SEGMENT
        st_att, ct_att = self.att_decoder(yt, (st_att, ct_att))

        qt = self.att_query(st_att).unsqueeze(1)
        if coverage_rate != 0:
            qt = qt + self.att_coverage(cov_att)
        at = self.att_weight(torch.tanh(kt_att + qt))
        at = F.softmax(at, dim=1) * tok_att_mask
        at = at / at.sum(dim=1, keepdim=True)
        vt = torch.sum(at*hs, dim=1)

        cov_att = cov_att + at

        p_wasa = self.att_ctx_classifier(vt) + self.att_hid_classifier(st_att)
        p_wasa = F.softmax(p_wasa, dim=-1)

        delta = self.final_gate(torch.cat([st_att, vt], dim=-1)).sigmoid()

        ## DECODE for DOCUMENT
        st_pnt, ct_pnt = self.pnt_decoder(yt, (st_pnt, ct_pnt))

        qt = self.pnt_query(st_pnt).unsqueeze(1)
        if coverage_rate != 0:
            qt = qt + self.pnt_coverage(cov_pnt)
        at = self.pnt_weight(torch.tanh(kt_pnt + qt))
        at = F.softmax(at, dim=1) * tok_pnt_mask
        at = at / at.sum(dim=1, keepdim=True)
        vt = torch.sum(at*hs, dim=1)

        cov_pnt = cov_pnt + at

        p_g = self.pnt_ctx_classifier(vt) + self.pnt_hid_classifier(st_pnt)
        p_g = F.softmax(p_g, dim=-1)

        p_c = torch.zeros(batch_size, self.word_size).cuda()
        w_idx = input.view(-1)
        b_idx = torch.arange(0, batch_size).unsqueeze(-1).expand(-1, doc_len*tok_len).contiguous().view(-1)
        s_idx = torch.arange(0, doc_len*tok_len).unsqueeze(-1).expand(-1, batch_size).transpose(0, 1).contiguous().view(-1)
        p_c[b_idx,w_idx] += at.squeeze(-1)[b_idx,s_idx]

        gamma = self.pnt_gate(torch.cat([st_pnt, vt], dim=-1)).sigmoid()
        p_pee = gamma * p_g + (1-gamma) * p_c

        pt = p_pee
        ps, ys = torch.topk(pt, k=20, dim=-1)

        count = 0
        for pt, yt in zip(ps[0], ys[0]):
          if count == k:
            break
          if yt == 1:
            continue
          if yt == eos and len(sequence) < 10:
            continue
          if len(sequence) >= 1:
            if sequence[-1] == yt:
              continue
          if len(sequence) >= 2:
            if tuple(sequence[-2:] + [yt]) in trigrams:
              continue
          if len(sequence) >= 3:
            if sequence[-3:-1] == sequence[-1:] + [yt]:
              continue
          count += 1
          new_instance = {
            'input': yt.unsqueeze(0),
            'states': (st_att, ct_att, st_pnt, ct_pnt),
            'coverage': (cov_att, cov_pnt),
            'sequence': sequence + [yt],
            'prob': prob + torch.log(pt),
            'prob_norm': (prob + torch.log(pt)) / len(sequence),
            'trigrams': trigrams + [tuple(sequence[-3:])]
          }
          new_beam.append(new_instance)
      beam = sorted(new_beam, key=lambda a: -a['prob_norm'])[:k]

    finished = sorted(finished, key=lambda a: -a['prob_norm'])[0]
    return torch.stack(finished['sequence'], dim=0)