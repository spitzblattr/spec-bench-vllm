import json
from decimal import Decimal
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

# 计算吞吐量加速比
def compute_throughput_ratios(baseline_data, improved_data):
    baseline_df = pd.DataFrame(baseline_data)
    improved_df = pd.DataFrame(improved_data)

    baseline_df['subtask'] = baseline_df['subtask'].apply(lambda x: x.split('_')[-1])
    improved_df['subtask'] = improved_df['subtask'].apply(lambda x: x.split('_')[-1])
    if 'THUCNews' in baseline_df['task'].values:
        # 去掉 'k'，将剩下的部分转换为整数并乘以 2
        baseline_df['subtask'] = baseline_df['subtask'].apply(lambda x: str(Decimal(str(float(x[:-1])*2)) )+"k")
        improved_df['subtask'] = improved_df['subtask'].apply(lambda x: str(Decimal(str(float(x[:-1])*2)) )+"k")

    merged_df = pd.merge(
        baseline_df[['subtask', 'max_concurrency', 'total_token_throughput']],
        improved_df[['subtask', 'max_concurrency', 'total_token_throughput']],
        on=['subtask', 'max_concurrency'],
        suffixes=('_baseline', '_improved')
    )

    merged_df['throughput_ratio'] = (
        merged_df['total_token_throughput_improved'] / merged_df['total_token_throughput_baseline']
    )

    heatmap_data = merged_df.pivot(
        index='max_concurrency',
        columns='subtask',
        values='throughput_ratio'
    )
    return heatmap_data


# 绘制热力图
def plot_heatmap(data, method_name, task_name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        data, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm",   
        cbar_kws={'label': 'Throughput Ratio'},
        vmin=0.0,  # 热力值最小值
        vmax=2.0   # 热力值最大值
    )
    plt.gca().invert_yaxis()  # 反转 y 轴顺序
    plt.title(f"Model:{method_name}  |  Task:{task_name} | num-spec-tokens=3")
    plt.xlabel("Context Length (prompt+output)")
    plt.ylabel("Max Concurrency")
    plt.yticks(rotation=0)  # 让纵轴标签水平显示
    plt.show(block=False)
    #plt.close()


if __name__ == "__main__":
    METHOD = ["7B", "7B-int4", "14B", "14B-int4", "72B", "72B-int4"]
    datasets_parent_dir = "./datasets/"
    result_parent_dir = f"./auto_test_results/online/"
    task_dataset_map = {
        "HighLight2News": datasets_parent_dir+"CNN-DailyMail-HighLights/",
        "News2HighLight": datasets_parent_dir+"CNN-DailyMail-News/",
        "THUCNews": datasets_parent_dir+"THUCNews/"
    }
    for i in METHOD:
        for j in list(task_dataset_map.keys()):
            # 加载两个 JSONL 文件
            improved_df = load_jsonl(result_parent_dir + f"{i}/Task_{j}/Table_MaxConcurrency.jsonl")
            baseline_df = load_jsonl(result_parent_dir + f"{i}-nospec/Task_{j}/Table_MaxConcurrency.jsonl")       
            # 计算加速比
            speedup_data = compute_throughput_ratios(baseline_df, improved_df)
            # 绘制热力图
            plot_heatmap(speedup_data, method_name=i, task_name=j)
            # plt.savefig会覆盖目录中已存在的同名图片
            plt.savefig(result_parent_dir + f"{i}/Task_{j}/heatmap_speedup_ratio.png") 
