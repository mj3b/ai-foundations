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

"""Inference function for generating text with pre-trained Gemma models.

This module provides a high-level function to prompt a Gemma model, generate
a text continuation, and retrieve the model's logits for the next token.
"""

from typing import Any

from ai_foundations.training import load_gemma
from gemma import gm
import jax.numpy as jnp
import numpy as np


def prompt_transformer_model(
    input_text: str,
    max_new_tokens: int = 10,
    model_name: str = "Gemma-1B",
    do_sample: bool = True,
) -> tuple[str, np.ndarray, Any]:
  """Generate text from a transformer model (Gemma) based on the input text.

  Args:
    input_text: The input prompt for the model.
    max_new_tokens: The maximum number of new tokens to generate.
    model_name: The name of the model to load. Supported options are
        'Gemma-1B' and 'Gemma-4B'.
    do_sample: Whether to use sampling for text generation (True for random
        sampling, False for greedy).

  Returns:
    output_text: The generated text, including the input text and the
        model's output.
    next_token_logits: Logits for the next token (probability distribution).
    tokenizer: The tokenizer used for encoding/decoding the text.

  Raises:
    ValueError: If the model_name is not recognized or supported.
  """

  if not isinstance(do_sample, bool):
    raise TypeError("do_sample must be a boolean value.")

  # Process for Gemma-based models.
  if model_name not in ["Gemma-1B", "Gemma-4B"]:
    raise ValueError(
        f"model_name=`{model_name}` is not supported. "
        "Supported options are 'Gemma-1B' and 'Gemma-4B'"
    )

  tokenizer, model, params = load_gemma(model_name)
  sampler = gm.text.Sampler(
      model=model,
      params=params,
      tokenizer=tokenizer,
  )

  if not do_sample:
    sampler_output_text = sampler.sample(
        input_text, max_new_tokens=max_new_tokens, sampling=gm.text.Greedy()
    )
  else:
    sampler_output_text = sampler.sample(
        input_text,
        max_new_tokens=max_new_tokens,
        sampling=gm.text.RandomSampling(),
    )

  # Convert the input text to tokens and apply the model to generate
  # predictions.
  prompt = tokenizer.encode(input_text, add_bos=True)
  prompt = jnp.asarray(prompt)
  out = model.apply(
      {"params": params},
      tokens=prompt,
      return_last_only=True,  # Only return the last token.
  )
  next_token_logits = out.logits
  output_text = input_text + sampler_output_text

  return output_text, next_token_logits, tokenizer
