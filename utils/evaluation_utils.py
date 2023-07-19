import pandas as pd

import itertools

from tqdm import tqdm
from typing import List, Dict

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, v_measure_score

from .data_utils import *
from text_clustering import TextClustering

def evaluate_classic_clustering(
    dataset: str,
    data_path: str,
    num_clasters: int,
    dim_reduction_type_list: List[str],
    n_components_list: List[int],
    clustering_type_list: List[str],
    random_state: int = 42,
    min_samples_list: List[int] = None,
    min_cluster_size_list: List[int] = None,
    bandwidth_list: List[float] = None,
    verbose: bool = False, 
    save_df: bool = False
) -> Tuple[pd.DataFrame, Dict[str, TextClustering]]:
    '''Run a series of exepriments on a given dataset. Save dataframe with results.
    
        :param dataset - name of the dataset (currently, 'banking' or 'demo')
        :param data_path - path to the directory with data and embeddings
        :param num_clasters - number of clusters to downsample from the dataset
        :param dim_reduction_type_list - list of dimentionality reduction types
        :param n_components_list - list of feature dimensionalities
        :param clustering_type_list - list of clustering types
        :param random_state - random state to be fixed for reproducibility
        :param min_samples_list - list of min_samples parameter values for grid search (if HDBSCAN is used)
        :param min_cluster_size_list - list of min_cluster_size parameter values for grid search (if HDBSCAN is used)
        :param bandwidth_list - list of bandwidth parameter values for grid search (if MeanShift is used)
        :param verbose - if True, print the the results
        :param save_df - if True, save the results into the "results/" folder
        :return a tuple of
            * res_df - dataframe with the results
            * dictionary with TextClustering models
    '''
    # load data
    if dataset == "banking":        
        data_all, clusters_all, embeds_t5_all = load_banking_data(data_path)
        base_embeds, _, base_data, base_clusters = sample_banking_clusters(
            dataframe=data_all,
            raw_embeds=embeds_t5_all,
            cluster_num_list=np.linspace(0, num_clasters - 1, num_clasters)
        )
    elif dataset == "demo":
        data, clusters, embeds = read_and_clean_sber_data(data_path)
        base_embeds, _, base_data, base_clusters = sample_sber_clusters(
            cluster_num=num_clasters,
            data_frame=data,
            embeds=embeds,
            ignore_other=True, # ignore cluster num 9 - "Другое"
            verbose=True
        )
    else:
        raise ValueError(f"Wrong dataset name {dataset}")
    
    inp_dim = base_embeds.shape[1]
    
    # save fitted models in a dictionary
    best_models_dict = dict()
    
    res_df = pd.DataFrame(columns=[
        "Dim reduction", "Clustering", "Best params",
        "Adjusted Rand", "Adjusted Mutual Info", "Silhouette Score", "Avg topic coherence",
        "Dim red time, sec", "Clustering time, sec"
    ])
    
    for dim_reduction_type, n_components, clustering_type in tqdm(
        itertools.product(dim_reduction_type_list, n_components_list, clustering_type_list), 
        total=len(dim_reduction_type_list) * len(n_components_list) * len(clustering_type_list)
    ):
        # skip
        if (
            (dim_reduction_type is None and n_components is not None) or
            (dim_reduction_type is not None and n_components is None)
           ):
            continue
            
        if verbose:
            print(f"dim_red: {dim_reduction_type}, n_comp: {n_components}, clustering: {clustering_type}")
                
        if clustering_type == "hdbscan":
            assert (min_samples_list is not None and min_cluster_size_list is not None), "Need to specify HDBSCAN parameters lists"
            
            best_metrics = dict()
            best_min_samples, best_min_cluster_size = 0, 0
            best_ars = -1e9
            best_model = None
            
            if verbose:
                print("Grid search for HDBSCAN parameters")
            for min_samples, min_cluster_size in tqdm(
                itertools.product(min_samples_list, min_cluster_size_list), 
                total=len(min_samples_list) * len(min_cluster_size_list)
            ):
                if verbose:
                    print(f"min_samples: {min_samples}, min_cluster_size: {min_cluster_size}")
                model = TextClustering(
                    n_clusters=num_clasters,
                    inp_dim=inp_dim,
                    train_dataset=base_embeds,
                    data_frame=base_data,
                    feat_dim=n_components,
                    kind="classic clustering",
                    dim_reduction_type=dim_reduction_type,
                    clustering_type=clustering_type,
                    min_samples=min_samples,
                    min_cluster_size=min_cluster_size,
                )

                model.fit(base_embeds)
                _, clusters = model.transform_and_cluster(base_embeds)
                
                try:
                    metrics = model.evaluate(use_true_clusters=True, verbose=verbose)
                except:
                    if verbose:
                        print(f"Not able to evaluate coherence. Skipping this parameters set.")
                    continue
                
                ars = metrics["adjusted_rand_score"]
                if ars > best_ars:
                    best_ars = ars
                    best_metrics = metrics
                    best_min_samples, best_min_cluster_size = min_samples, min_cluster_size
                    best_model = model
                    
            times = model.times
            best_params = [f"min_s = {best_min_samples}", f"min_cl_s = {best_min_cluster_size}"]
            
            if verbose:
                for k, v in metrics.items():
                    print(f"{k}: {np.round(v, 4)}")
                
        elif clustering_type == "mean_shift":
            assert bandwidth_list is not None, "Need to specify MeanShift parameter lists"
            
            best_metrics = dict()
            best_bandwidth = 0
            best_ars = -1e9
            best_model = None
            
            if verbose:
                print("Grid search for MeanShift parameters")
                
            for bandwidth in tqdm(bandwidth_list):
                if verbose:
                    print(f"bandwidth: {bandwidth}")
                
                model = TextClustering(
                    n_clusters=num_clasters,
                    inp_dim=inp_dim,
                    train_dataset=base_embeds,
                    data_frame=base_data,
                    feat_dim=n_components,
                    kind="classic clustering",
                    dim_reduction_type=dim_reduction_type,
                    clustering_type=clustering_type,
                    bandwidth=bandwidth
                )

                model.fit(base_embeds)
                _, clusters = model.transform_and_cluster(base_embeds)
                
                try:
                    metrics = model.evaluate(use_true_clusters=True, verbose=verbose)
                except:
                    if verbose:
                        print(f"Not able to evaluate coherence. Skipping this parameters set.")
                    continue
                
                ars = metrics["adjusted_rand_score"]
                if ars > best_ars:
                    best_ars = ars
                    best_metrics = metrics
                    best_bandwidth = bandwidth
                    best_model = model
            
            times = model.times
            best_params = [f"bandwidth = {best_bandwidth}"]
            
            if verbose:
                for k, v in metrics.items():
                    print(f"{k}: {np.round(v, 4)}")
            
        else:
            best_model = TextClustering(
                n_clusters=num_clasters,
                inp_dim=inp_dim,
                train_dataset=base_embeds,
                data_frame=base_data,
                feat_dim=n_components,
                kind="classic clustering",
                dim_reduction_type=dim_reduction_type,
                clustering_type=clustering_type,
            )

            best_model.fit(base_embeds)
            _, clusters = best_model.transform_and_cluster(base_embeds)
            
            try:
                best_metrics = best_model.evaluate(use_true_clusters=True, verbose=verbose)
            except:
                if verbose:
                    print(f"Not able to evaluate coherence. Skipping this parameters set.")
                continue
            
            times = best_model.times
            best_params = ["-"]
            
        if verbose:
            print("-" * 80)
        
        if dim_reduction_type is None:
            times["dim_red"] = 0.0
        
        dim_rediction = (
            (dim_reduction_type if dim_reduction_type is not None else "-") +
            (f", feat_dim = {n_components}" if n_components is not None else "")
        )
        
        best_models_dict[f"{clustering_type}_{dim_rediction}_{best_params}"] = best_model
           
        new_row = pd.DataFrame.from_dict({
                "Dim reduction": [dim_rediction],
                "Clustering": [clustering_type],
                "Best params": [" ".join(best_params)],
                "Adjusted Rand": [best_metrics["adjusted_rand_score"]],
                "Adjusted Mutual Info": [best_metrics["adjusted_mutual_info_score"]],
                "Silhouette Score": [best_metrics["silhouette_score"]],
                "Avg topic coherence": [best_metrics["average_topic_coherence"]],
                "Dim red time, sec": [times["dim_red"]],
                "Clustering time, sec": [times["clust"]]
            })
        res_df = pd.concat([res_df, new_row], ignore_index=True)
    
    res_df = res_df.round(4)
    
    if save_df:
        res_df.to_csv(f"results/classic_results_{dataset}{num_clasters}.csv")
        
    return res_df, best_models_dict

def clustering_similarity(
    best_models_dict: Dict[str, TextClustering], sim_metric: str = "rand_score"
) -> pd.DataFrame:
    '''Evaluate similarity matrix for clustering algorithms.
    
        :param best_models_dict - dictionary with TextClustering models
        :param sim_metric - type of metric for similarity evaluation ("rand_score", "mutual_info" and "v_measure" are available)
        :return np.array of shape (n_models, n_models) with clustering similarity values
    '''
    
    def enumerated_product(*args):
        '''Utility for iteration.'''
        yield from zip(itertools.product(*(range(len(x)) for x in args)), itertools.product(*args))
    
    if sim_metric == "rand_score":
        criterion = adjusted_rand_score
    elif sim_metric == "mutual_info":
        criterion = adjusted_mutual_info_score
    elif sim_metric == "v_measure":
        criterion = v_measure_score
    else:
        raise ValueError(f"Not supported similarity metric '{sim_metric}'.")
    
    n_models = len(best_models_dict)
    sim_matrix = np.zeros((n_models, n_models))

    for (i_1, i_2), (model_1, model_2) in tqdm(
        enumerated_product(best_models_dict.values(), best_models_dict.values()), 
        total=n_models*n_models
    ):
        sim_matrix[i_1, i_1] = 1.0000

        if i_2 >= i_1:
            continue

        _, clusters_1 = model_1.transform_and_cluster(model_1.train_dataset)
        _, clusters_2 = model_2.transform_and_cluster(model_2.train_dataset)

        sim = np.round(criterion(clusters_1, clusters_2), 4).item()
        sim_matrix[i_1, i_2] = sim
        sim_matrix[i_2, i_1] = sim
    
    return sim_matrix