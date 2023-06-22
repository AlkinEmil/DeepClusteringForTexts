import pandas as pd
from keybert import KeyBERT


def get_topics_for_english(texts, labels, n_clusters):
    data_train = pd.DataFrame({'text': texts, 'cluster': labels})
    kw_model = KeyBERT()
    keywords_all = []
    for cluster_num in range(5):
        texts_cluster = data_train[data_train["cluster"] == cluster_num]["text"].to_list()
        texts_cluster = " ".join(texts_cluster)
        keyword_pairs = kw_model.extract_keywords(
            texts_cluster,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            use_mmr=True,
            diversity=0.3
        )[:3]
        keywords = [pair[0] for pair in keyword_pairs]
        keywords_all.append(keywords)
    return keywords_all