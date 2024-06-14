
import re
import numpy as np
import random as rand
import math
import torch
import pandas as pd

from main import TXT_FILENAME, VOCAB

class DataTools:

  @staticmethod
  def mask_characters(word, num_chars_to_mask=-1, to_mask=False):
      if type(word) == float:
          print(word)
          word = str(word)
      if not to_mask:
          to_mask = rand.random()*0.9
      if num_chars_to_mask == -1:
          num_chars_to_mask = int(max([1, int(to_mask*len(set(word)))]))

      # Ensure num_chars_to_mask is within the valid range
      if num_chars_to_mask < 1 or num_chars_to_mask > len(set(word)):
          raise ValueError("Number of characters to mask is out of range")

      # Select unique characters from the word
      unique_chars = list(set(word))

      # Randomly choose num_chars_to_mask characters to mask
      chars_to_mask = rand.sample(unique_chars, num_chars_to_mask)

      # Create the masked word
      masked_word = ''.join(['_' if char in chars_to_mask else char for char in word])

      cannot_guess = sorted(list(set("abcdefghijklmnop") - (set(word))))

      random_guessed_letters = rand.sample(cannot_guess, min(6, int(rand.random()*len(cannot_guess))+1))

      label = DataTools.get_label(word, masked_word, to_mask)
      for c in random_guessed_letters:
          label[VOCAB[c]] = 0#-1

      return masked_word + "%" + ''.join(sorted(random_guessed_letters)), label


  @staticmethod
  def get_dictionary_df(txt_filename=TXT_FILENAME):
      return pd.read_csv(filepath_or_buffer=txt_filename, encoding="utf8", names=['words'])
  
  @staticmethod
  def get_label(word, masked, to_mask=False):
      guessable = set(word) - (set(masked))
      freqs = [0]*len(VOCAB)

      for c in word:
          freqs[VOCAB[c]] += 1 + (freqs[VOCAB[c]]) if c in guessable else 0  

      for c in range(len(freqs)):
          if freqs[c] == 0:
              freqs[c] = -torch.inf

      return torch.softmax(torch.tensor(freqs), dim=0)*math.sqrt(1-to_mask)*2


  @staticmethod
  def get_dataset_from_df(df):
      df['transformed'] = df['words'].apply(DataTools.mask_characters)
      df['src'] = df['transformed'].apply(lambda x : x[0])
      df['tgt'] = df['transformed'].apply(lambda x : x[1])

      src, tgt = list(df['src']), list(df['tgt'])
      return src, tgt


class DictionaryDataset(torch.utils.data.Dataset):
    def __init__(self, txt_filename=TXT_FILENAME, truncate=False) -> None:
        # load data + preprocess
        df = DataTools.get_dictionary_df(txt_filename)[:truncate] if truncate else DataTools.get_dictionary_df(txt_filename)
        self.src, self.tgt = DataTools.get_dataset_from_df(df)
        self.src = list(self.src)
        self.tgt = self.tgt

    def __getitem__(self, idx) -> torch.Tensor:
        return self.src[idx], self.tgt[idx]

    def __len__(self):
        return len(self.src)