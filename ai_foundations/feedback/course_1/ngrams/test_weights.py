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

"""A utility function to test a learner's n-gram model weights.

This module provides a function to validate if a user can
correctly identify the probability weights for a given context in a trigram
model.
"""

from typing import Dict, Iterable
from ai_foundations.feedback.utils import render_feedback


def test_weights(
    trigram_model: Dict[str, Dict[str, float]], learner_weights: Iterable[float]
) -> None:
  """Tests if the learner's list of weights is correct for a given context.

  Args:
    trigram_model: The ngram model dictionary.
    learner_weights: An iterable of floats provided by the learner to be tested.
  """

  hint = """
    This is very similar to getting the candidate words, but this time you need
    the probability values. Use the <code>.values()</code> method on the
    dictionary of candidates.
    """

  context = "looking for"

  try:
    if context not in trigram_model.keys():
      raise KeyError(
          "Sorry, your answer is not correct.",
          f"The context '{context}' does not exist in the trigram model.",
      )

    if list(learner_weights) != list(trigram_model[context].values()):
      raise ValueError(
          "Sorry, your answer is not correct.",
          "The list of weights does not match the expected values.",
      )

  except (KeyError, ValueError) as e:
    render_feedback(e, hint)

  else:
    print("✅ Nice! Your answer looks correct.")
