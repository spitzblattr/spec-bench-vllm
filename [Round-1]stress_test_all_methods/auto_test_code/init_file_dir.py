import os
import json

def create_directory_structure(base_path):

    online_subfolders = ["eagle", "medusa", "sps", "pld", "vanilla"]
    task_subfolders = [
        "Task_ShareGPT",
        "Task_SpecBench_conversation",
        "Task_SpecBench_math_reasoning",
        "Task_SpecBench_qa",
        "Task_SpecBench_rag",
        "Task_SpecBench_summarization",
        "Task_SpecBench_translation",
        "Task_SpecBench_All",
    ]
    # json_files = ["Table_MaxConcurrency.jsonl", "Table_QPS.jsonl"]
    json_files = ["Table_MaxConcurrency.jsonl"]

    # 创建根目录
    auto_test_results_path = os.path.join(base_path, "auto_test_results")
    online_path = os.path.join(auto_test_results_path, "online")

    for folder in online_subfolders:
        folder_path = os.path.join(online_path, folder)

        for task_folder in task_subfolders:
            task_folder_path = os.path.join(folder_path, task_folder)
            os.makedirs(task_folder_path, exist_ok=True)  

            # 创建空白 JSON 文件
            for json_file in json_files:
                json_file_path = os.path.join(task_folder_path, json_file)
                with open(json_file_path, "w") as f:
                    pass  # 不知为何不open一下就不会创建空白文件

if __name__ == "__main__":
    # 在该根目录下生成存放测试结果的子文件夹
    create_directory_structure("/PATH/TO/YOUR/[Round-1]stress_test_all_methods")
    print("创建测试结果文件目录完成")
