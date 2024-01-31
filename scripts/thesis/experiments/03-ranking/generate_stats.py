import pandas as pd

if __name__ == '__main__':
    results_all_ensembles_from_nn = pd.read_csv('perf_results_nearest_neighbor-retrain_all_ensembles_from_nn'
                                                '.csv')
    top_k = 12
    sorted_df = results_all_ensembles_from_nn.sort_values(['task_id', 'original_task_disagreement'],
                                                          ascending=False)
    grouped = sorted_df.groupby('task_id')
    top_k_per_group = grouped.head(top_k)

    sorted_k = top_k_per_group.sort_values(['task_id', 'accuracy'], ascending=False)
    grouped_k = sorted_k.groupby('task_id')
    best_from_group = grouped_k.head(1)

    best_from_group.to_csv(f'perf_results_nearest_neighbor-mixed-acc-doublefault_rank_top_{top_k}-new.csv',
                           header=True,
                           index=False)
