import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from generate_embed import rank_corpus
import random
import numpy as np
import nltk

nltk.download("punkt", quiet=True)
nltk.download("brown", quiet=True)
nltk.download("wordnet", quiet=True)
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
import nltk

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
import string
import pke
import traceback
from flashtext import KeywordProcessor


def load_model(model="ramsrigouthamg/t5_squad_v1"):
    question_model = T5ForConditionalGeneration.from_pretrained(
        "ramsrigouthamg/t5_squad_v1"
    )
    question_tokenizer = T5Tokenizer.from_pretrained("ramsrigouthamg/t5_squad_v1")
    return question_model, question_tokenizer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


def postprocesstext(content):
    final = ""
    for sent in sent_tokenize(content):
        sent = sent.capitalize()
        final = final + " " + sent
    return final


def get_nouns_multipartite(content):
    out = []
    try:
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=content)
        #    not contain punctuation marks or stopwords as candidates.
        pos = {"PROPN", "NOUN"}
        # pos = {'PROPN','NOUN'}
        stoplist = list(string.punctuation)
        stoplist += ["-lrb-", "-rrb-", "-lcb-", "-rcb-", "-lsb-", "-rsb-"]
        stoplist += pke.lang.stopwords.get("en")
        extractor.candidate_selection(pos=pos)
        # 4. build the Multipartite graph and rank candidates using random walk,
        #    alpha controls the weight adjustment mechanism, see TopicRank for
        #    threshold/method parameters.
        extractor.candidate_weighting(alpha=1.1, threshold=0.75, method="average")
        keyphrases = extractor.get_n_best(n=15)

        for val in keyphrases:
            out.append(val[0])
    except:
        out = []
        traceback.print_exc()

    return out


def get_keywords(originaltext, num_prompt=5):
    keywords = get_nouns_multipartite(originaltext)
    # print ("keywords unsummarized: ",keywords)
    keyword_processor = KeywordProcessor()
    for keyword in keywords:
        keyword_processor.add_keyword(keyword)

    keywords_found = keyword_processor.extract_keywords(originaltext)
    keywords_found = list(set(keywords_found))
    # print ("keywords_found in summarized: ",keywords_found)

    important_keywords = []
    for keyword in keywords:
        if keyword in keywords_found:
            important_keywords.append(keyword)

    return important_keywords[:num_prompt]


def get_question(context, answer, model, tokenizer):
    text = "context: {} answer: {}".format(context, answer)
    encoding = tokenizer.encode_plus(
        text,
        max_length=384,
        pad_to_max_length=False,
        truncation=True,
        return_tensors="pt",
    )
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        early_stopping=True,
        num_beams=5,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        max_length=72,
    )

    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]

    Question = dec[0].replace("question:", "")
    Question = Question.strip()
    return Question


def gen_text(ranked_doc):
    doc = []
    for d in ranked_doc.values():
        doc.append(d["Sentence"])

    text = " ".join(doc)
    text = postprocesstext(text)
    text = text.strip()

    return text


if __name__ == "__main__":
    query = "How was Napolean Buonaparte realted to French Revolution and France?"
    context = "Napoleon Bonaparte (1769-1821) was a military and political leader of France who rose to prominence during the French Revolution and became the emperor of France from 1804 to 1814. He is widely regarded as one of the greatest military commanders in history, and his campaigns and conquests during the early 19th century greatly expanded the territory of France and reshaped the political landscape of Europe. Some of his most famous military campaigns include the Italian Campaign, the Egyptian Campaign, and the Napoleonic Wars. Despite his military successes, Napoleon's reign was marked by controversy and his rule ended in defeat and exile."
    ranked_doc = rank_corpus(query, context, "word2vec")
    text = gen_text(ranked_doc)

    num_prompt = 5
    imp_keywords = get_keywords(text, num_prompt)
    query_tokens = []
    prompt = []

    model, tokenizer = load_model()
    for key in imp_keywords:
        ques = get_question(text, key, model, tokenizer)
        prompt.append(ques)
        query_tokens.append(key.capitalize())

    print(query_tokens)
    print(prompt)
