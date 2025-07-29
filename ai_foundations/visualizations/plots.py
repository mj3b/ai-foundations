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

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import plotly.express as px


def plot_next_token(
    probs_or_logits: np.ndarray,
    tokenizer: Any,
    prompt: str,
    keep_top: int = 30
):
  """Plot the probability distribution of the next tokens.

  This function generates a bar plot showing the top `keep_top` tokens by
  probability.

  Args:
    probs_or_logits: The raw logits output by the model or the probability
        distribution for the next token prediction.
    tokenizer: The tokenizer used to decode token IDs to human-readable text.
    prompt: The input prompt used to generate the next token predictions.
    keep_top: The number of top tokens to display in the plot.
  """

  if np.isclose(probs_or_logits.sum(), 1):
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
  tokens = [repr(tokenizer.decode(i.item())) for i in indices]

  # Create the bar plot using Plotly.
  fig = px.bar(x=tokens, y=probs)

  # Customize the plot layout.
  fig.update_layout(
      title=(
          "Probability distribution of next "
          f'tokens given the prompt="{prompt}"'
      ),
      xaxis_title="Tokens",
      yaxis_title="Probability",
  )

  # Display the plot.
  fig.show()
