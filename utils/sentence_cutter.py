import nltk
import jieba
import sudachipy
import langid
nltk.download('punkt')
langid.set_languages(['en', 'zh', 'ja'])

def split_text_into_sentences(text):
    if langid.classify(text)[0] == "en":
        sentences = nltk.tokenize.sent_tokenize(text)

        return sentences
    elif langid.classify(text)[0] == "zh":
        sentences = []
        segs = jieba.cut(text, cut_all=False)
        segs = list(segs)
        start = 0
        for i, seg in enumerate(segs):
            if seg in ["。", "！", "？", "……"]:
                sentences.append("".join(segs[start:i + 1]))
                start = i + 1
        if start < len(segs):
            sentences.append("".join(segs[start:]))

        return sentences
    elif langid.classify(text)[0] == "ja":
        sentences = []
        tokenizer = sudachipy.Dictionary().create()
        tokens = tokenizer.tokenize(text)
        current_sentence = ""

        for token in tokens:
            current_sentence += token.surface()
            if token.part_of_speech()[0] == "補助記号" and token.part_of_speech()[1] == "句点":
                sentences.append(current_sentence)
                current_sentence = ""

        if current_sentence:
            sentences.append(current_sentence)

        return sentences

    raise RuntimeError("It is impossible to reach here.")