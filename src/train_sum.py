import argparse
import utils
import numpy as np
import torch
from denoisesum import DenoiseSum


def pretrain(args, pretrain_mode):
  word_dict = utils.get_lm_dict(args.train_file)

  x_train = utils.lm_data(args.train_file, word_dict)
  x_dev = utils.lm_data(args.dev_file, word_dict)

  model = Mustard(args.word_size,
                  args.word_dim,
                  args.hidden_dim,
                  args.disc_size)
  model.cuda()

  optimizer = torch.optim.Adagrad(model.parameters(), lr=0.1, initial_accumulator_value=0.1)
  best_loss = 1000

  if os.path.exists(args.pretrain_file):
    best_point = torch.load(args.pretrain_file)
    model.load_state_dict(best_point['state_dict'])
    optimizer.load_state_dict(best_point['optimizer'])
  
  if pretrain_mode == 'encoder':
    forward = model.encoder_lm
  else:
    forward = model.decoder_lm

  eval_count = args.eval_every
  stop_count = args.stop_after

  for epoch in range(args.num_epoch):
    if stop_count <= 0:
      break

    shuffle_indices = np.random.permutation(np.arange(len(x_train)))
    x_shuffle = np.array(x_train)[shuffle_indices]

    losses = []
    train_iterator = tqdm(range(0, len(x_shuffle), batch_size))
    for i in train_iterator:
      if stop_count <= 0:
        train_iterator.close()
        break

      model.train()

      x_batch, x_mask = utils.pad(x_shuffle[i:i+batch_size])
      x_batch = torch.tensor(x_batch).cuda()
      x_mask = torch.tensor(x_mask).float().cuda()

      batch_loss = forward(x_batch, x_mask)
      losses.append(batch_loss.item())

      batch_loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), 3)
      optimizer.step()
      optimizer.zero_grad()

      eval_count -= len(x_batch)
      if eval_at <= 0:
        with torch.no_grad():
          shuffle_indices = np.random.permutation(np.arange(len(x_dev)))[:2000]
          x_dev = np.array(x_dev)[shuffle_indices]

          train_loss = np.mean(losses)
          dev_loss = []

          for j in tqdm(range(0, len(x_dev), batch_size)):
            model.eval()

            x_batch, x_mask = utils.pad(x_dev[j:j+batch_size])
            x_batch = torch.tensor(x_batch).cuda()
            x_mask = torch.tensor(x_mask).float().cuda()

            batch_loss = forward(x_batch, x_mask)
            dev_loss.append(batch_loss.item())

          dev_loss = np.mean(dev_loss)
          if best_loss >= dev_loss:
            best_loss = dev_loss
            stop_count = args.stop_after
            torch.save({
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict()
            }, args.pretrain_file)
          else:
            stop_count -= 1

          tqdm.write("Epoch: %d, Batch: %d, Train Loss: %.4f, Dev Loss: %.4f" % (epoch, i, train_loss, dev_loss))
          losses = []
          eval_count = args.eval_every



def train(args):
  word_dict = utils.get_lm_dict(args.train_file)

  train_data = utils.noisy_data(args.train_file, word_dict, count=100000, dict_type='word')
  dev_data = utils.clean_data(args.dev_file, word_dict, dict_type='word')

  x1_train, x2_train, c_train, y_train = train_data
  x_train = [x1+x2 for x1,x2 in zip(x1_train, x2_train)]
  x_dev, c_train, y_dev = test_data
  if dataset == 'rotten':
    m_dev = utils.get_movies(file_dir + '/test.json')

  rev_word_dict = {value:key for key, value in word_dict.items()}

  model = Mustard(args.word_size,
                  args.word_dim,
                  args.hidden_dim,
                  args.disc_size)
  model.cuda()

  optimizer = torch.optim.Adam(model.parameters())
  best_bleu = 0
  best_loss = 1000

  if os.parh.exists(args.pretrain_file):
    best_point = torch.load(args.pretrain_file)
    model.load_partial_state_dict(best_point['state_dict'])
  if os.path.exists(args.model_file):
    best_point = torch.load(args.model_file)
    model.load_state_dict(best_point['state_dict'])
    optimizer.load_state_dict(best_point['optimizer'])
    best_bleu = best_point['dev_bleu']

  eval_count = args.eval_every
  stop_count = args.stop_after

  for epoch in range(args.num_epoch):
    if stop_count <= 0:
      break

    shuffle_indices = np.random.permutation(np.arange(len(x_train)))
    x_shuffle = np.array(x_train)[shuffle_indices]
    y_shuffle = np.array(y_train)[shuffle_indices]
    c_shuffle = np.array(c_train)[shuffle_indices]

    losses = []
    train_iterator = tqdm(range(0, len(x_shuffle), args.batch_size))
    for i in train_iterator:
      if stop_count <= 0:
        train_iterator.close()
        break

      model.train()

      x_batch = x_shuffle[i:i+batch_size]
      y_batch = y_shuffle[i:i+batch_size]
      c_batch = c_shuffle[i:i+batch_size]

      x_pad = utils.hier_pad(x_batch, 
                             att_end=args.no_instance//2, 
                             pnt_start=args.no_instance//2,
                             max_tok_length=200)
      x_batch, doc_att_mask, doc_pnt_mask, tok_att_mask, tok_pnt_mask = x_pad

      y_batch, y_mask = utils.pad(y_batch)

      x_batch = torch.tensor(x_batch).cuda()
      doc_att_mask = torch.tensor(doc_att_mask).float().cuda()
      doc_pnt_mask = torch.tensor(doc_pnt_mask).float().cuda()
      tok_att_mask = torch.tensor(tok_att_mask).float().cuda()
      tok_pnt_mask = torch.tensor(tok_pnt_mask).float().cuda()
      y_batch = torch.tensor(y_batch).cuda()
      y_mask = torch.tensor(y_mask).float().cuda()
      c_batch = torch.tensor(c_batch).float().cuda()

      p_batch, loss = model(x_batch
                            doc_att_mask,
                            doc_pnt_mask,
                            tok_att_mask,
                            tok_pnt_mask,
                            y_batch,
                            y_mask,
                            category=c_batch,
                            coverage_rate=args.coverage_rate)

      losses.append(loss.item())

      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), 3)
      nan_check = False:
      for param in model.parameters():
        if param.grad is not None:
          if torch.isnan(param.grad.sum()):
            nan_check = True
            break
      if not nan_check:
        optimizer.step()
        optimizer.zero_grad()

      eval_count -= len(x_batch)
      if eval_count <= 0:
        with torch.no_grad():
          train_loss = np.mean(losses)

          pred_sums = []
          gold_sums = []
          dev_loss = []
          printing = 5
          for j in tqdm(range(0, len(x_dev), 1)):
            model.eval()

            x_batch = x_dev[j:j+1]
            y_batch = y_dev[j:j+1]
            if args.dataset == 'rotten':
              m_batch = m_dev[j:j+1]
            else:
              m_batch = ['MOV']

            x_pad = utils.hier_pad(x_batch,
                                   att_end=args.no_instance//2,
                                   pnt_start=args.no_instance//2,
                                   max_tok_length=200)
            x_batch, doc_att_mask, doc_pnt_mask, tok_att_mask, tok_pnt_mask = x_pad

            y_batch, y_mask = utils.pad(y_batch)

            x_batch = torch.tensor(x_batch).cuda()
            doc_att_mask = torch.tensor(doc_att_mask).float().cuda()
            doc_pnt_mask = torch.tensor(doc_pnt_mask).float().cuda()
            tok_att_mask = torch.tensor(tok_att_mask).float().cuda()
            tok_pnt_mask = torch.tensor(tok_pnt_mask).float().cuda()
            y_batch = torch.tensor(y_batch).cuda()
            y_mask = torch.tensor(y_mask).float().cuda()
            c_batch = None

            p_batch, loss = model(x_batch,
                                  doc_att_mask,
                                  doc_pnt_mask,
                                  tok_att_mask,
                                  tok_pnt_mask,
                                  y_batch,
                                  y_mask,
                                  category=c_batch,
                                  coverage_rate=args.coverage_rate)

            dev_loss.append(loss.item())

            y_batch = y_batch.cpu().detach().numpy()
            p_batch = p_batch.argmax(-1).cpu().detach().numpy()
            for y, p, m in zip(y_batch, p_batch, m_batch):
              y = list([int(yy) for yy in y])
              p = list([int(pp) for pp in p])
              try:
                y = y[1:y.index(eos)]
              except:
                pass
              try:
                p = p[:p.index(eos)]
              except:
                pass
              y_text = ' '.join([rev_word_dict[yy] for yy in y])
              p_text = ' '.join([rev_word_dict[pp] for pp in p])
              pred_sums.append(p_text)
              gold_sums.append(y_text)
              if printing:
                printing -= 1
                tqdm.write('gold: %s' % y_text)
                tqdm.write('pred: %s' % p_text)
                tqdm.write('-----------------------------------------')

        gold_sums = [[gold] for gold in gold_sums]
        dev_bleu = corpus_bleu(gold_sums, pred_sums)
        dev_loss = np.mean(dev_loss)

        if dev_bleu >= best_bleu:
          tqdm.write('updating model...')
          best_loss = dev_loss
          best_bleu = dev_bleu
          stop_count = stop_after
          torch.save({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'dev_bleu': dev_bleu,
            'dev_loss': dev_loss
          }, args.model_file)
        else:
          stop_count -= 1

        tqdm.write("Epoch: %d, Batch: %d, Train Loss: %.4f, Dev Loss: %.4f, Dev BLEU: %.4f"
                   % (epoch, i, train_loss, dev_loss, dev_bleu))
        losses = []
        eval_count = args.eval_every




if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('-dataset', default='rotten', type=str)
  parser.add_argument('-no_instance', default=40, type=int)

  parser.add_argument('-batch_size', default=8, type=int)
  parser.add_argument('-word_dim', default=300, type=int)
  parser.add_argument('-hidden_dim', default=512, type=int)
  parser.add_argument('-disc_size', default=50, type=int)

  parser.add_argument('-num_epoch', default=20, type=int)
  parser.add_argument('-eval_every', default=2500, type=int)
  parser.add_argument('-stop_after', default=40, type=int)

  parser.add_argument('-train_file', default='train.synthetic.json', type=str)
  parser.add_argument('-dev_file', default='dev.json', type=str)
  parser.add_argument('-test_file', default='test.json', type=str)

  parser.add_argument('-pretrain_file', default='pretrain.model', type=str)
  parser.add_argument('-model_file', default='denoisesum.model', type=str)

  parser.add_argument('-sos', default=2, type=int)
  parser.add_argument('-eos', default=3, type=int)

  parser.add_argument('-coverage_rate', default=0, type=int)