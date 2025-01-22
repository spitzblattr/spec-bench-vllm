import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_jsonl_to_df(filepath):
    with open(filepath, 'r') as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)


def calculate_speedup_ratio(improved_df, baseline_df):
    merged_df = pd.merge(
        improved_df,
        baseline_df,
        on="max_concurrency",
        suffixes=('_improved', '_baseline')
    )
    # 计算加速比
    merged_df['speedup_ratio'] = (
        merged_df['total_token_throughput_improved'] / merged_df['total_token_throughput_baseline']
    )
    return merged_df[['NUM_SPEC_TOKENS_improved', 'max_concurrency', 'speedup_ratio']]

def plot_heatmap(data, method_name, task_name):
    heatmap_data = data.pivot(
        index='max_concurrency', 
        columns='NUM_SPEC_TOKENS_improved', 
        values='speedup_ratio'
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm",   
        cbar_kws={'label': 'Speedup Ratio'},
        vmin=0.0,  
        vmax=2.0   
    )
    plt.gca().invert_yaxis()  
    plt.title(f"Method:{method_name}  |  Task:{task_name}")
    plt.xlabel("num-spec-tokens")
    plt.ylabel("Max Concurrency")
    plt.yticks(rotation=0)  
    plt.show(block=False)
    #plt.close()


if __name__ == "__main__":
    METHOD = ["eagle", "medusa", "pld", "sps"]
    datasets_parent_dir = "./PATH/TO/YOUR/datasets/"
    result_parent_dir = f"./PATH/TO/YOUR/auto_test_results/online/"
    task_dataset_map = {
        "ShareGPT": datasets_parent_dir+"sharegpt_processed_1024.json",
        "SpecBench_conversation": datasets_parent_dir+"spec-bench-conversation.jsonl",
        "SpecBench_math_reasoning": datasets_parent_dir+"spec-bench-math-reasoning.jsonl",
        "SpecBench_qa": datasets_parent_dir+"spec-bench-qa.jsonl",
        "SpecBench_summarization": datasets_parent_dir+"spec-bench-summarization.jsonl",
        "SpecBench_rag": datasets_parent_dir+"spec-bench-rag.jsonl",   
        "SpecBench_translation": datasets_parent_dir+"spec-bench-translation.jsonl",
        "SpecBench_All": datasets_parent_dir+"spec-bench-all.jsonl",
    }
    for i in METHOD:
        for j in list(task_dataset_map.keys()):
            # 加载两个 JSONL 文件
            improved_df = load_jsonl_to_df(result_parent_dir + f"{i}/Task_{j}/Table_MaxConcurrency.jsonl")
            baseline_df = load_jsonl_to_df(result_parent_dir + f"vanilla/Task_{j}/Table_MaxConcurrency.jsonl")       
            speedup_data = calculate_speedup_ratio(improved_df, baseline_df)       
            plot_heatmap(speedup_data, method_name=i, task_name=j)
            plt.savefig(result_parent_dir + f"{i}/Task_{j}/heatmap_speedup_ratio.png") 
            # 找到最大的加速比及其对应的 NUM_SPEC_TOKENS 和 max_concurrency
            max_speedup_row = speedup_data.loc[speedup_data['speedup_ratio'].idxmax()]
            max_entry = {
                "method": i,
                "task": j,
                "num_spec_tokens": int(max_speedup_row['NUM_SPEC_TOKENS_improved']),
                "max_concurrency": int(max_speedup_row['max_concurrency']),
                "speedup_ratio": float(max_speedup_row['speedup_ratio'])
            }
            # 追加写入到最大值 JSONL 文件
            output_file = result_parent_dir + "max_speedup.jsonl"
            with open(output_file, "a") as f:
                f.write(json.dumps(max_entry) + "\n")