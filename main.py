import string
import unicodedata
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import trainer
import model



def read_book(file_path):
    """Read the content of the book from the file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def write_unique_words(words, output_path):
    """Write each unique word to the output file, one word per line."""
    with open(output_path, 'w', encoding='utf-8') as file:
        for word in words:
            file.write(word + '\n')

def normalize_text(text):
    """Normalize text to remove diacritics and special characters."""
    # Normalize text to NFD (Normalization Form Decomposition)
    normalized_text = unicodedata.normalize('NFD', text)
    # Remove diacritics
    text_without_diacritics = ''.join(
        char for char in normalized_text if unicodedata.category(char) != 'Mn'
    )
    return text_without_diacritics

def process_text(text):
    """Process the text to normalize and extract unique words."""
    # Normalize text
    text = normalize_text(text)
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Split text into words
    words = text.split()
    # Filter out non-alphabetic words and get unique words
    unique_words = set(word for word in words if word.isalpha())
    return unique_words

def main(input_file, output_file):
    """Main function to read, process, and write the text."""
    # Read the book
    text = read_book(input_file)
    # Process the text to get unique words
    unique_words = process_text(text)
    # Write unique words to the output file
    write_unique_words(unique_words, output_file)


def get_liks(word, transformer):
    prediction = transformer(torch.tensor([list(TEXT_TRANSFORM_LAMBDA(word))]))
    return prediction[0]

def guess_letter(masked, excluded, transformer):
    print(masked)
    liks = get_liks(masked, transformer=transformer)
    reverse_voca_lookup = {VOCAB[k]:k for k in VOCAB}
    pairs = {reverse_voca_lookup[i]:liks[i] for i in range(len(liks))}
    for c in excluded:
        pairs.pop(c)
    pairs = dict(sorted(pairs.items(), key=lambda item: item[1], reverse=True))
    lets = [p[0] for p in pairs]
    return lets[0], lets, pairs


def play_game(model, word):
    if type(model) == str:
        model = torch.load(model)
    masked = '_'*len(word)

    wrongs = 0
    guessed = set()
    wrong_guesses = set()

    while(wrongs < 6):
        in_word = set(masked) - {'_'}
        guess, _, _ = guess_letter(masked, excluded=guessed.union(in_word), transformer=model)
        masked = ''.join([word[i] if word[i] == guess else masked[i] for i in range(len(word))]) + "%" + ''.join(sorted(list(wrong_guesses)))
        print(guess, masked)

        guessed.add(guess)
        if guess not in set(word):
            wrongs+=1
            wrong_guesses.add(guess)

        if masked[0:masked.find("%")]==word:
            print("YOU WON!")
            print(wrongs, " wrong guesses")
            break

input_file = './book.txt'
output_file = './words.txt'
main(input_file, output_file)

TOKENIZE_LAMBDA = lambda x : list(x)
VOCAB = {c : i+2 for i, c in enumerate(list('abcdefghijklmnopqrstuvwxyz'))}
VOCAB['_'] = 1 # mask
VOCAB['$'] = 0 # pad
VOCAB['%'] = len(VOCAB) # sep
PAD_IDX = VOCAB['$']

VOCAB_TRANSFORM_LAMBDA = lambda toks : torch.tensor(np.vectorize(lambda x : VOCAB[x])(toks) )
TOKENIZE_LAMBDA = lambda x : list(x)
TEXT_TRANSFORM_LAMBDA = lambda x : VOCAB_TRANSFORM_LAMBDA(TOKENIZE_LAMBDA(x))
TXT_FILENAME = output_file

NUM_EPOCHS = 15
torch.manual_seed(0)

SRC_VOCAB_SIZE = len(VOCAB)
EMB_SIZE = 26
NHEAD = 2
FFN_HID_DIM = 26
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 10

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":

    transformer = model.MaskedLmTransformer(
        num_encoder_layers=NUM_ENCODER_LAYERS,
        emb_size=EMB_SIZE,
        nhead=NHEAD,
        vocab_size=SRC_VOCAB_SIZE,
        dim_feedforward=FFN_HID_DIM,
        dropout= 0.1
    )

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    # The model was already trained over Google Colab

    # Uncomment this code if you wish to train the model again 
    trainer.Trainer.train(transformer, optimizer, loss_fn, NUM_EPOCHS, BATCH_SIZE)

    # Uncomment this code to save the model into a file after training
    torch.save(transformer, f="/content/drive/My Drive/saved_model")

    play_game(transformer, "laceration")
