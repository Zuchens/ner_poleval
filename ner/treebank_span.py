from nltk import TreebankWordTokenizer, PunktSentenceTokenizer


class TreebankSpanTokenizer(TreebankWordTokenizer):

    def __init__(self):
        self._word_tokenizer = TreebankWordTokenizer()

    def span_tokenize(self, text):
        ix = 0
        for word_token in self.tokenize(text):
            ix = text.find(word_token, ix)
            end = ix + len(word_token)
            yield ix, end, word_token
            ix = end

    def tokenize(self, text):
        return self._word_tokenizer.tokenize(text)

# class PunktSentenceSpanTokenizer(PunktSentenceTokenizer):
#
#     def __init__(self):
#         self._sentence_tokenizer = PunktSentenceTokenizer()
#
#     def sentence_span_tokenize(self, text):
#         ix = 0
#         for word_token in self.tokenize(text):
#             ix = text.find(word_token, ix)
#             end = ix + len(word_token)
#             yield ix, end
#             ix = end
#
#     def tokenize(self, text):
#         return self._sentence_tokenizer.tokenize(text)
