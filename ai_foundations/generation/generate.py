# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Text generation utilities for a trained Keras model.

This module provides functions for autoregressive text generation, supporting
both greedy decoding and random sampling methods.
"""

from typing import Any

import keras
from keras import ops
import numpy as np


def sampling(probs: np.ndarray) -> int:
  """Sample a token index from the predicted next token probability.

  Args:
    probs: The probability distribution of predicted next token.

  Returns:
    The index of the sampled token.
  """
  return np.random.choice(np.arange(len(probs)), p=probs)


def greedy_decoding(probs: np.ndarray) -> int:
  """Select the token index from the predicted next token probability.

  Args:
    probs: The probability distribution of predicted next token.

  Returns:
    The index of the token with the highest probability.
  """
  predicted_index = np.argmax(probs).astype(int)
  return predicted_index


def generate_text(
    start_prompt: str,
    n_tokens: int,
    model: keras.Model,
    tokenizer: Any,
    pad_token_id: int = 0,
    do_sample: bool = False,
) -> tuple[str, list[np.ndarray]]:
  """Generate text based on a starting prompt using a trained model.

  Args:
    start_prompt: The initial prompt to start the generation.
    n_tokens: The number of tokens to generate after the prompt.
    model: The trained model to use for text generation.
    tokenizer: The tokenizer to encode and decode text.
    pad_token_id: The token ID used for padding.
    do_sample: Whether to sample from the distribution or use greedy decoding.

  Returns:
    The generated text after the prompt.
  """
  max_length = model.layers[0].output.shape[1]

  # Tokenize the starting prompt.
  start_tokens = tokenizer.encode(start_prompt)

  # Generate tokens.
  tokens_generated = start_tokens + []
  probs: list[np.ndarray] = []
  for _ in range(n_tokens):
    pad_len = max_length - len(start_tokens)
    sample_index = len(start_tokens) - 1
    if pad_len < 0:
      # Truncate the input sequence to fit the max context length.
      x = start_tokens[:max_length]
      sample_index = max_length - 1
    elif pad_len > 0:
      x = (
          start_tokens + [pad_token_id] * pad_len
      )  # Pad the input sequence.
    else:
      x = start_tokens

    x = np.array([x])
    # Get predictions from the model.
    y = model.predict(x, verbose="0")

    # Apply softmax to convert logits to probabilities.
    probabilities = ops.softmax(y, axis=-1).numpy()  # type: ignore

    probs.append(probabilities[0][sample_index])

    # Use greedy decoding or sampling based on the flag.
    if not do_sample:
      sample_token = greedy_decoding(probabilities[0][sample_index])
    else:
      sample_token = sampling(probabilities[0][sample_index])

    tokens_generated.append(sample_token)
    start_tokens.append(sample_token)

  # Convert tokens back to text.
  generated_text = tokenizer.decode(tokens_generated)
  generated_text = generated_text.replace(tokenizer.decode([pad_token_id]), "")

  return generated_text, probs
