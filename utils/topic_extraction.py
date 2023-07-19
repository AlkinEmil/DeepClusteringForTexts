import pandas as pd

from typing import List, Tuple, Dict

from keybert import KeyBERT
from gensim.models import CoherenceModel
import gensim.corpora as corpora


def get_topics_for_cluster(
    texts_cluster: List[str],
    kw_model: KeyBERT,
    language: str,
    diversity: float = 0.3,
    keyphrase_ngram_range: Tuple[int] = (1, 2),
    phrases_num: int = 3
) -> List[str]:
    '''Extract keywords for texts in one cluster using KeyBert.
    
        :param texts_cluster - list with texts for key word extraction
        :param kw_model - KeyBERT model for key word extraction
        :param language - language of texts
        :param diversity - parameter regulating key phrases diversity
        :param keyphrase_ngram_range - length of the key phrases for extraction
        :param phrases_num - number of key phrases to be extracted
        :return list of key phrases for texts
    '''
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


def get_topics(
    data_frame: pd.DataFrame,
    kw_model: KeyBERT = None,
    language: str = "english",
    diversity: float = 0.3,
    keyphrase_ngram_range: Tuple[int] = (1, 2),
    phrases_num: int = 3
) -> Dict[int, List[str]]:
    '''Get cluster-topics dictionary for predicted clusters.
    
        :param data_frame - pd.DataFrame with texts (column "text") for topic extraction; should contain "pred_cluster" column
        :param kw_model - KeyBERT model for key word extraction
        :param language - language of texts
        :param diversity - parameter regulating key phrases diversity
        :param keyphrase_ngram_range - length of the key phrases for extraction
        :param phrases_num - number of key phrases to be extracted
        :return cluster-topics dictionary for predicted clusters
    '''
    if kw_model is None and language == "english":
        # use default sentence BERT model for English
        kw_model = KeyBERT()
    elif kw_model is None and language == "russian":
        # use Russian sentence BERT model
        kw_model = KeyBERT("DeepPavlov/rubert-base-cased-sentence")
    else:
        raise ValueError("Wrong language for topic extraction.")
        
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


def compute_coherence_for_clusters(
    topics_cluster_dict: Dict[int, List[str]], data_frame: pd.DataFrame, verbose: bool = False
) -> Dict[int, float]:
    '''Compute coherence metric for extracted topics using gensim.CoherenceModel().
    
        :param topics_cluster_dict - dictionary with key words of predicted clusters
        :param data_frame - pd.DataFrame with texts (column "text") for topic extraction; should contain "pred_cluster" column
        :param verbose - if True, print the results
        :return cluster-coherence dictionary for predicted clusters
    '''
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