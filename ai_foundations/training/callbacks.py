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

"""A Keras callback to generate sample text during model training.

This module defines the TextGenerator callback, which can be used with
`model.fit()` to monitor a language model's progress by generating and
printing sample text at the end of specified epochs.
"""

from typing import Any, Dict, List

import keras
from keras import ops
import numpy as np


class TextGenerator(keras.callbacks.Callback):
  """A callback to generate text from a trained model.

    1. Feed an initial prompt to the model.
    2. Predict probabilities for the next token.
    3. Sample the next token and add it to the input for the next prediction.

  Attributes:
    max_tokens: Number of tokens to generate.
    start_tokens: Token indices for the initial prompt.
    tokenizer: The tokenizer used to decode generated token indices.
    pad_token_id: The padding token ID.
    print_every: Print the generated text every `print_every` epochs.
    **callback_kwargs: Any additional keyword arguments.
  """

  def __init__(
      self,
      max_tokens: int,
      start_tokens: List[int],
      tokenizer: Any,
      pad_token_id: int = 0,
      print_every: int = 1,
      **callback_kwargs: Dict[str, Any],
  ):
    super().__init__(**callback_kwargs)

    self.max_tokens = max_tokens
    self.start_tokens = start_tokens
    self.tokenizer = tokenizer
    self.print_every = print_every
    self.pad_token_id = pad_token_id  # ID for padding token.

  def greedy_decoding(self, probs: np.ndarray) -> int:
    """Select the token index with the highest probability.

    Args:
      probs: The probability distribution of next token prediction.

    Returns:
      The index of the predicted token with the highest probability.
    """
    predicted_index = np.argmax(probs).astype(int)
    return predicted_index

  def sampling(self, probs: np.ndarray) -> int:
    """Sample a token index from the predicted next token probability.

    Args:
      probs: The probability distribution of predicted next token.

    Returns:
      The index of the sampled token.
    """

    return np.random.choice(np.arange(len(probs)), p=probs)

  def on_epoch_end(
      self, epoch: int, logs: Dict[str, Any] | None = None
  ) -> None:
    """Generate and print text after each epoch based on starting tokens.

    Args:
      epoch: The current epoch number.
      logs: Logs from the training process.
    """

    if self.model is None:
      return

    max_length = self.model.layers[0].output.shape[1]
    # Make a copy of the start tokens.
    start_tokens = list(self.start_tokens)
    if (epoch + 1) % self.print_every != 0:
      return

    num_tokens_generated = 0
    tokens_generated: list[int] = []

    while num_tokens_generated < self.max_tokens:
      pad_len = max_length - len(start_tokens)
      sample_index = len(start_tokens) - 1

      # Handle padding to ensure the sequence is of the correct length.
      if pad_len < 0:
        x = start_tokens[:max_length]
        sample_index = max_length - 1
      elif pad_len > 0:
        x = start_tokens + [self.pad_token_id] * pad_len
      else:
        x = start_tokens

      x = np.array([x])
      y = self.model.predict(x, verbose=0)

      # Convert logits to probabilities using softmax.
      probabilities = ops.softmax(y, axis=-1).numpy()

      sample_token = self.sampling(probabilities[0][sample_index])

      tokens_generated.append(sample_token)
      start_tokens.append(sample_token)
      num_tokens_generated = len(tokens_generated)

    # Combine the starting tokens with the generated tokens.
    output_tokens = self.start_tokens + tokens_generated
    output_tokens = list(map(int, output_tokens))

    # Decode and print the generated text.
    txt = self.tokenizer.decode(output_tokens)
    print("Generated text:\n", txt, "\n")
