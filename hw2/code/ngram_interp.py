import math

from lm import LangModel
from ngram import Ngram
from typing import List

import numpy as np


class InterpNgram(LangModel):
    """Interpolated N-gram Language Model with backoff"""

    def __init__(self, ngram_size: int, alpha: float, llambda: float, **kwargs):
        super().__init__(**kwargs)
        assert 0 < alpha < 1
        assert 0 <= llambda
        assert 0 < ngram_size and isinstance(ngram_size, int)

        if ngram_size == 2:
            self.backoff_model = Ngram(1, llambda=llambda, **kwargs)
        else:
            self.backoff_model: InterpNgram = InterpNgram(ngram_size - 1, alpha, llambda=llambda, **kwargs)

        self.alpha = alpha
        self.model = Ngram(ngram_size, llambda=llambda, **kwargs)
        self.ngram_size = ngram_size

    @property
    def name(self):
        return f"interp_{self.ngram_size}-gram"

    def fit_sentence(self, sentence: List[str]):
        for i, word_i in enumerate(sentence):
            self.incr_word(sentence[:i], word_i)

    def incr_word(self, context: List[str], word: str):
        self.model.incr_word(context, word)
        self.backoff_model.incr_word(context, word)

    def cond_logprob(self, word: str, context: List[str]) -> float:
        context = self.model.get_context(context)
        cur_model_word_context = self.model.counts[context].get(word, None)
        cur_context_count = self.model.counts_totals.get(context, None)
        if cur_model_word_context is None: cur_model_word_context = 0

        if cur_context_count is None:
            cur_model_prob = math.exp(self.backoff_model.cond_logprob(word, context))
        else:
            cur_model_prob = math.exp(self.model.cond_logprob(word, context))

        log_prob = (self.alpha * cur_model_prob) + ((1-self.alpha) * math.exp(self.backoff_model.cond_logprob(word, context)))
        log_prob = math.log(log_prob) if log_prob != 0 else float('-inf')
        return log_prob
