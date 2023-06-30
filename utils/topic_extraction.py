import pandas as pd

from keybert import KeyBERT
from gensim.models import CoherenceModel
import gensim.corpora as corpora


def get_topics_for_cluster(texts_cluster, kw_model, language, diversity=0.3, keyphrase_ngram_range=(1, 2), phrases_num=3):
    texts_cluster = " ".join(texts_cluster)
    
    keyword_pairs = kw_model.extract_keywords(
        texts_cluster,
        keyphrase_ngram_range=keyphrase_ngram_range,
        stop_words="english",
        use_mmr=True,
        diversity=diversity,
        )[:phrases_num]
    
    keywords = [pair[0] for pair in keyword_pairs]
    
    return keywords


def get_topics(data_frame, kw_model=None, language="english", diversity=0.3, keyphrase_ngram_range=(1, 2), phrases_num=3):
    if kw_model is None and language == "english":
        kw_model = KeyBERT()
    elif kw_model is None and language == "russian":
        #kw_model = KeyBERT("DeepPavlov/rubert-base-cased-sentence")
        kw_model = KeyBERT("ai-forever/ruBert-base")
        
    topics_cluster_dict = {}
    
    assert "pred_cluster" in data_frame.columns, "Predicted clusters are not in the DataFrame"
    
    for cluster_num in data_frame["pred_cluster"].unique():
        texts_cluster = data_frame[data_frame["pred_cluster"] == cluster_num]["text"].to_list()
        keywords = get_topics_for_cluster(
            texts_cluster,
            kw_model=kw_model,
            language=language,
            diversity=diversity,
            keyphrase_ngram_range=keyphrase_ngram_range,
            phrases_num=phrases_num
        )

        topics_cluster_dict[cluster_num] = keywords
    return topics_cluster_dict


def compute_coherence_for_clusters(topics_cluster_dict, data_frame, verbose=False):    
    topics_list = []
    for topic_phrases_list in topics_cluster_dict.values():
        curr_topic_list = []
        for topic in topic_phrases_list:
            curr_topic_list.extend(topic.split())

        topics_list.append(curr_topic_list)

    all_texts, tokens_list = [], []
    cluster_nums = list(topics_cluster_dict.keys())
    
    assert "pred_cluster" in data_frame.columns, "Predicted clusters are not in the DataFrame"

    for cluster_num in cluster_nums:
        texts_cluster = data_frame[data_frame["pred_cluster"] == cluster_num]["text"].to_list()
        doc_cluster = " ".join(texts_cluster)
        all_texts.append(doc_cluster.split())
        
        curr_tokens = list(doc_cluster.split())
        tokens_list.append(curr_tokens)
        
    id2word = corpora.Dictionary(tokens_list)
    corpus = [id2word.doc2bow(text) for text in tokens_list]
    
    topics_coh_list = CoherenceModel(
        topics=topics_list,
        texts=all_texts,
        dictionary=id2word,
        corpus=corpus,
        coherence='c_v',
    ).get_coherence_per_topic()
    
    topics_coh_dict = {list(topics_cluster_dict.keys())[i]: topics_coh_list[i] for i in range(len(cluster_nums))}
    
    if verbose:
        for k, v in topics_coh_dict.items():
            print(f"Cluster num {k}, topics: {topics_cluster_dict[k]}, Coherence = {v}")
    
    return topics_coh_dict