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

"""A plotting utility function for visualizing language model predictions.

This module contains a function to create a bar chart of the most probable next
tokens, given a model's output logits or probabilities.
"""

from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import plotly.express as px


def plot_next_token(
    probs_or_logits: np.ndarray | Dict[str, float],
    prompt: str,
    keep_top: int = 30,
    tokenizer: Optional[Any] = None
):
  """Plot the probability distribution of the next tokens.

  This function generates a bar plot showing the top `keep_top` tokens by
  probability.

  Args:
    probs_or_logits: The raw logits output by the model or the probability
        distribution for the next token prediction. Can also be a dictionary as
        returned by an n-gram model.
    prompt: The input prompt used to generate the next token predictions.
    keep_top: The number of top tokens to display in the plot.
    tokenizer: The tokenizer used to decode token IDs to human-readable text.

  Returns:
    Displays a plot showing the probability distribution of the top tokens.
  """

  if isinstance(probs_or_logits, dict):
    # Extract probabilities from n-gram dictionary.
    probs = jnp.array(list(probs_or_logits.values()))
  elif np.isclose(probs_or_logits.sum(), 1):
    probs = probs_or_logits
  else:
    # Apply softmax to logits to get probabilities.
    probs = jax.nn.softmax(probs_or_logits)

  # Select the top `keep_top` tokens by probability.
  indices = jnp.argsort(probs)

  # Reverse to get highest probabilities first.
  indices = indices[-keep_top:][::-1]

  # Get the probabilities and corresponding tokens.
  probs = probs[indices].astype(np.float32)

  if tokenizer is not None:
    # Decode indices using tokenizer.
    tokens = [repr(tokenizer.decode(index.item())) for index in indices]
  elif isinstance(probs_or_logits, dict):
    # Extract tokens from n-gram dictionary.
    tokens = list(probs_or_logits.keys())
  else:
    # Return the raw indices if no decoding information is supplied.
    tokens = indices

  # Create the bar plot using Plotly.
  fig = px.bar(x=tokens, y=probs)

  # Customize the plot layout.
  fig.update_layout(
      title=(
          f'Probability distribution of next tokens given the prompt="{prompt}"'
      ),
      xaxis_title="Tokens",
      yaxis_title="Probability",
  )

  # Display the plot.
  fig.show()
