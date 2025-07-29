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

"""A utility function to test a learner's n-gram model solution.

This module provides a function to validate if a user can
correctly identify candidate words for a given context in a trigram model.
"""

from typing import Dict, Iterable
from ai_foundations.feedback.utils import render_feedback


def test_candidate_words(
    trigram_model: Dict[str, Dict[str, float]], candidate_words: Iterable[str]
) -> None:
  """Tests if the learner correctly identifies candidate words from a model.

  This function checks if the provided `candidate_words` set correctly
  matches the keys for the context 'looking for' within the provided
  trigram model.

  Args:
    trigram_model: A dictionary representing the ngram model, mapping contexts
        to a dictionary of candidate words and their probabilities.
    candidate_words: A set of strings representing the learner's
        identified candidate words.
  """

  hint = """
    The ngram model is a dictionary where each context maps to another dictionar
    of candidate words.<br>
    To get a list of candidate words for a specific context, put the context in
    the model as <code>ngram_model[context]</code> and then use the
    <code>.keys()</code> method.
    """

  context = "looking for"

  try:
    if context not in trigram_model.keys():
      raise KeyError(
          "Sorry, your answer is not correct.",
          f"The context '{context}' does not exist in the trigram model.",
      )

    if candidate_words != trigram_model[context].keys():
      raise ValueError(
          "Sorry, your answer is not correct.",
          "Your set of words does not match the expected candidates.",
      )

  except (KeyError, ValueError) as e:
    render_feedback(e, hint)

  else:
    print("✅ Nice! Your answer looks correct.")
