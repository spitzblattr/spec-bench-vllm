import os
import json

def create_directory_structure(base_path):

    online_subfolders = ["7B-nospec", "7B", "7B-int4", "7B-int4-nospec", "14B-nospec", "14B", "14B-int4", "14B-int4-nospec","72B-nospec", "72B", "72B-int4", "72B-int4-nospec"]
    task_subfolders = [  
        "Task_HighLight2News",    # 给摘要生成新闻   
        "Task_News2HighLight",        # 给新闻生成摘要  
        "Task_THUCNews",        # 润色
    ]
    json_files = ["Table_MaxConcurrency.jsonl"]

    # 创建根目录
    auto_test_results_path = os.path.join(base_path, "auto_test_results")
    online_path = os.path.join(auto_test_results_path, "online")

    for folder in online_subfolders:
        folder_path = os.path.join(online_path, folder)

        for task_folder in task_subfolders:
            task_folder_path = os.path.join(folder_path, task_folder)
            os.makedirs(task_folder_path, exist_ok=True)  

            for json_file in json_files:
                json_file_path = os.path.join(task_folder_path, json_file)
                with open(json_file_path, "w") as f:
                    pass  # 不知为何不open一下就不会创建空白文件

if __name__ == "__main__":
    # 在该根目录下生成存放测试结果的子文件夹
    create_directory_structure("/PATH/TO/YOUR/[Round-2]stress_test_sps_only")
    print("创建测试结果文件目录完成")
