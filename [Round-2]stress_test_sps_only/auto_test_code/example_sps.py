import os
import subprocess
import requests
import time
from globals import *

# ————————————————————————————————————————————————————————————————————————————————————
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5" 
PORT = "7778"      
METHOD = "72B"      
# ---------------------------------------
model_path = "/mnt/public/models/Qwen2.5-72B-Instruct-AWQ"
speculative_model_path = "/mnt/public/models/Qwen2.5-0.5B-Instruct-AWQ"
result_parent_dir = f"/home/spitzblattr/[Round-2]stress_test_sps_only/auto_test_results/online/{METHOD}/"
# ————————————————————————————————————————————————————————————————————————————————————

def run_vllm_server(num_spec_tokens):
    vllm_server_command = [
        python_exe_path, "-m", "vllm.entrypoints.openai.api_server",
        "--host", "0.0.0.0",
        "--port", PORT,
        "--model", model_path,
        "--dtype", "bfloat16",
        "--tensor-parallel-size", "3",                          
        "--speculative-model", speculative_model_path,
        "--gpu-memory-utilization", "0.9",
        "--use-v2-block-manager",
        "--num-speculative-tokens", num_spec_tokens,
        "--disable-log-requests"
    ]
    p = subprocess.Popen(vllm_server_command)
    return p

def run_stress_test_client(task_name, dataset_path, max_concurrency=None, qps=None, context_length=None):
    dataset_name = "cnn_dailymail_news" if "News2HighLight" in task_name else "cnn_dailymail_highlights" if "HighLight2News" in task_name else "THUCNews"
    num_prompts = 48 if "8k" in context_length else 48 if "4k" in context_length else 96

    client_command = [
        python_exe_path, online_benchmark_script_path,
        "--backend", "openai-chat",
        "--base-url", f"http://localhost:{PORT}",  
        "--endpoint", "/v1/chat/completions",       
        "--dataset-name", dataset_name,     
        "--dataset-path", dataset_path,
        "--model", model_path,      
        "--disable-tqdm",      
        "--save-result",       
        "--result-dir", result_parent_dir+"Task_"+task_name+"/",   
        "--seed", "12345",
        "--num-prompts", str(num_prompts), 
        "--num-spec-tokens", str(num_spec_tokens),   
    ]
    if max_concurrency is not None:
        client_command.extend(["--max-concurrency", max_concurrency])
        client_command.extend(["--request-rate", "inf"])
        client_command.extend(["--result-filename", "Table_MaxConcurrency.jsonl"])
    elif qps is not None:
        client_command.extend(["--request-rate", qps])
        client_command.extend(["--result-filename", "Table_QPS.jsonl"])

    subprocess_test = subprocess.Popen(client_command,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True)
    stdout_data, stderr_data = subprocess_test.communicate()
    print("Subprocess STDOUT:", stdout_data)
    print("Subprocess STDERR:", stderr_data)
    print("Return code:", subprocess_test.returncode)

    return subprocess_test


if __name__ == "__main__":
     
    for num_spec_tokens in ["3"]:      
        p = run_vllm_server(num_spec_tokens)
        time.sleep(240)
        try:
            r = requests.get(f"http://0.0.0.0:{PORT}/v1/models")
            if r.status_code == 200:
                print(f"============ 开启 vllm openai server, 当前参数: num_spec_tokens = {num_spec_tokens} ============")
        except requests.exceptions.ConnectionError:
            raise TimeoutError("服务启动失败")

        for task in list(task_dataset_map.keys()):
            for max_concurrency in ["1","4","8","12"]:  
                for i, context_length in enumerate(context_lengths_list := ["0.25k","0.5k", "1k", "2k", "4k", "8k"]):            
                    print(f"\n\t🌟\tcurrent test: task={task} | context_length={context_length} |  max_concurrency={max_concurrency}\n")
                    
                    real_dataset_path = None
                    if task == "News2HighLight":
                        real_dataset_path = task_dataset_map[task]+"cnn_dailymail_"+context_length+".jsonl"
                    elif task == "HighLight2News":
                        real_dataset_path = task_dataset_map[task]+"cnn_dailymail_highlights_"+context_length+".jsonl"
                    elif task == "THUCNews":
                        if context_length == "0.25k":
                            real_dataset_path = task_dataset_map[task]+"thucnews_"+"0.125k"+".jsonl"
                        else:
                            real_dataset_path = task_dataset_map[task]+"thucnews_"+context_lengths_list[i-1]+".jsonl"

                    subprocess_test = run_stress_test_client(task_name=task, dataset_path=real_dataset_path,
                                                                max_concurrency=max_concurrency, qps=None,
                                                                context_length=context_length)

                    subprocess_test.terminate() 
                    subprocess_test.wait()    
        
        # ------ 关闭服务 ------
        print(f"正在关闭 vllm 服务...")
        p.terminate()  
        p.wait()     
        print("vllm 服务已关闭，准备进入下一个循环参数...\n")

    print(f"🌟 🌟 🌟 {METHOD} 全部测试完成")