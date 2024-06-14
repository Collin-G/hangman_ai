from torch.utils.data import DataLoader
from timeit import default_timer as timer
from torch.nn.utils.rnn import pad_sequence
import torch

from datatools import DictionaryDataset
from main import DEVICE, PAD_IDX, TEXT_TRANSFORM_LAMBDA

class Trainer:
  @staticmethod
  def get_train_test_iter():
    total_data_iter = DictionaryDataset()#truncate=100)
    train_size = int(0.8 * len(total_data_iter))
    test_size = len(total_data_iter) - train_size
    train_iter, test_iter = torch.utils.data.random_split(total_data_iter, [train_size, test_size])
    return train_iter, test_iter

  @staticmethod
  def train_epoch(model, optimizer, loss_fn,train_iter, batch_size):
      model.train()
      losses = 0
      train_dataloader = DataLoader(train_iter, batch_size=batch_size, collate_fn=Trainer.collate_fn)

      for src, tgt in train_dataloader:
          src = src.to(DEVICE)
          tgt = tgt.to(DEVICE)#, dtype=torch.float)

          optimizer.zero_grad()

          logits = model(src)

          loss = loss_fn(logits, tgt)
          loss.backward()

          optimizer.step()
          losses += loss.item()

      return losses/len(list(train_dataloader))

  @staticmethod
  def evaluate(model, test_iter, loss_fn,batch_size):
      model.eval()
      losses = 0

      val_dataloader = DataLoader(test_iter, batch_size=batch_size, collate_fn=Trainer.collate_fn)


      for src, tgt in val_dataloader:
          src = src.to(DEVICE)
          tgt = tgt.to(DEVICE)#, dtype=torch.float)

          logits = model(src)

          loss = loss_fn(logits, tgt)
          losses += loss.item()

      return losses / len(list(val_dataloader))
  
  @staticmethod
  def train(model, optimizer, loss_fn,num_epochs, batch_size):
    train_iter, test_iter = Trainer.get_train_test_iter()
    for epoch in range(1, num_epochs+1):
        start_time = timer()
        train_loss = Trainer.train_epoch(model, optimizer, loss_fn, train_iter, batch_size)
        end_time = timer()
        val_loss = Trainer.evaluate(model, test_iter, loss_fn, batch_size)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
  
  @staticmethod
  def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(TEXT_TRANSFORM_LAMBDA(src_sample))
        tgt_batch.append((list(tgt_sample)))
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    return src_batch, torch.tensor(tgt_batch)