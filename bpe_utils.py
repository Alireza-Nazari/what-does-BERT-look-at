"""Going from BERT's bpe (Byte pair encoding) tokenization to word-level tokenization.


... Just for Simpler Analysis ...

"""

import utils
from bert import tokenization

import numpy as np


def tokenize_and_align(tokenizer, words, cased):
  """Given already-tokenized text (as a list of strings), returns a list of
  lists where each contains BERT-tokenized tokens for the
  correponding word."""

  words = ["[CLS]"] + words + ["[SEP]"]
  basic_tokenizer = tokenizer.basic_tokenizer
  tokenized_words = []
  for word in words:
    # Basic Cleaning & Prep: Convert to Unicode, remove control chars/extra spaces.
    word = tokenization.convert_to_unicode(word)
    word = basic_tokenizer._clean_text(word)

    if word == "[CLS]" or word == "[SEP]":
      word_toks = [word]
    else:
      # Casing & Punctuation Split: Handle lowercasing/accent stripping for uncased models,
      # then split words like "hello." into ["hello", "."].
      if not cased:
        word = word.lower()
        word = basic_tokenizer._run_strip_accents(word)
      word_toks = basic_tokenizer._run_split_on_punc(word)

    tokenized_word = []
    for word_tok in word_toks:
      # WordPiece Tokenization: Break down words into subword units (e.g., "tokenization" -> "token", "##ization").
      tokenized_word += tokenizer.wordpiece_tokenizer.tokenize(word_tok)
    tokenized_words.append(tokenized_word)

  i = 0
  word_to_tokens = []
  for word_subtokens_list in tokenized_words:
    tokens = []
    # Index Alignment: Map original words to the global indices of their corresponding BERT subword tokens.
    # This creates a crucial lookup table for converting attention.
    for _ in word_subtokens_list:
      tokens.append(i)
      i += 1
    word_to_tokens.append(tokens)
  assert len(word_to_tokens) == len(words)

  return word_to_tokens


def get_word_word_attention(token_token_attention, words_to_tokens,
                            mode="first"):
  """Convert token-token attention to word-word attention (when tokens are
  derived from words using something like byte-pair encodings)."""

  word_word_attention = np.array(token_token_attention)
  not_word_starts = []
  for word_indices in words_to_tokens:
    not_word_starts += word_indices[1:]

  # Consolidate Received Attention (Columns): Sum attention *to* all subword tokens of a word
  # and assign it to the first subword token's column. Then delete redundant subword columns.
  for word_indices in words_to_tokens:
    word_word_attention[:, word_indices[0]] = word_word_attention[:, word_indices].sum(axis=-1)
  word_word_attention = np.delete(word_word_attention, not_word_starts, -1)

  # Aggregate Given Attention (Rows): Combine attention *from* subword tokens
  # of a word into the first subword token's row, using the specified `mode` (first, mean, or max).
  for word_indices in words_to_tokens:
    if mode == "first":
      pass # Already using the first subword's attention
    elif mode == "mean":
      word_word_attention[word_indices[0]] = np.mean(word_word_attention[word_indices], axis=0)
    elif mode == "max":
      word_word_attention[word_indices[0]] = np.max(word_word_attention[word_indices], axis=0)
      word_word_attention[word_indices[0]] /= word_word_attention[word_indices[0]].sum() # Normalize max mode
    else:
      raise ValueError("Unknown aggregation mode", mode)
  # Finally, delete redundant subword rows, completing the word-level conversion.
  word_word_attention = np.delete(word_word_attention, not_word_starts, 0)

  return word_word_attention


def make_attn_word_level(data, tokenizer, cased):
  # Iterate and Transform: Loop through dataset features, align words to tokens,
  # then convert all token-level attention maps to word-level using the above functions.
  # The `assert` acts as a quick sanity check for token counts.
  for features in utils.logged_loop(data):
    words_to_tokens = tokenize_and_align(tokenizer, features["words"], cased)
    assert sum(len(word_indices) for word_indices in words_to_tokens) == len(features["tokens"])
    features["attns"] = np.stack([[
        get_word_word_attention(attn_head, words_to_tokens)
        for attn_head in layer_attns] for layer_attns in features["attns"]])