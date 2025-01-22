import os
import subprocess
import requests
import time
from globals import *

# ————————————————————————————————————————————————————————————————————————————————————
os.environ["CUDA_VISIBLE_DEVICES"] = "5" 
PORT = "7779"       
METHOD = "pld"      # "eagle", "medusa", "pld", "sps", "vanilla"
# ---------------------------------------
model_path = "/PATH/TO/YOUR/Qwen2.5-7B-Instruct"
result_parent_dir = f"/PATH/TO/YOUR/stress_test/auto_test_results/online/{METHOD}/"
# ————————————————————————————————————————————————————————————————————————————————————

def run_vllm_server(num_spec_tokens, ngram_max):
    vllm_server_command = [
        python_exe_path, "-m", "vllm.entrypoints.openai.api_server",
        "--host", "0.0.0.0",
        "--port", PORT,
        "--model", model_path,
        "--dtype", "bfloat16",
        "--tensor-parallel-size", "1",                          
        "--speculative-model", "[ngram]",
        "--ngram-prompt-lookup-max", ngram_max,
        "--gpu-memory-utilization", "0.9",
        "--use-v2-block-manager",
        "--num-speculative-tokens", num_spec_tokens,
        "--disable-log-requests",
    ]
    p = subprocess.Popen(vllm_server_command)
    return p

def run_stress_test_client(task_name, dataset_path, max_concurrency=None, qps=None, num_spec_tokens=None):
    dataset_name = "sharegpt" if "ShareGPT" in task_name else "specbench"
    num_prompts = 512 if "ShareGPT" in task_name else 480 if "SpecBench_All" in task_name else 80

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
        # ------
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

    for num_spec_tokens in ["1","2","3","4","5"]:
        p = run_vllm_server(num_spec_tokens=num_spec_tokens, ngram_max=num_spec_tokens)
        time.sleep(240)
        try:
            r = requests.get(f"http://0.0.0.0:{PORT}/v1/models")
            if r.status_code == 200:
                print(f"============ 开启 vllm openai server, 当前参数: num_spec_tokens = {num_spec_tokens} ============")
        except requests.exceptions.ConnectionError:
            raise TimeoutError("服务启动失败")

        for task in list(task_dataset_map.keys()):
            for max_concurrency in ["1","4","8","12","16","32","64","128"]:                   
                print(f"\n\t🌟\tcurrent test: num_spec_tokens={num_spec_tokens} | task={task} | max_concurrency={max_concurrency}\n")
                subprocess_test = run_stress_test_client(task_name=task, dataset_path=task_dataset_map[task],
                                                            max_concurrency=max_concurrency, qps=None,
                                                            num_spec_tokens=num_spec_tokens)
                subprocess_test.terminate() 
                subprocess_test.wait()  

            '''
            for qps in ["1","2","4","8","12"]:    
                print(f"\n\t🌟\tcurrent test: num_spec_tokens={num_spec_tokens} | task={task} | qps={qps}\n")
                subprocess_test = run_stress_test_client(task_name=task, dataset_path=task_dataset_map[task],
                                                            max_concurrency=None, qps=qps,
                                                            num_spec_tokens=num_spec_tokens)   
                subprocess_test.terminate() 
                subprocess_test.wait()     
            '''
        
        print(f"正在关闭 vllm 服务...")
        p.terminate()  
        p.wait()       
        print("vllm 服务已关闭，准备进入下一个循环参数...\n")

    print(f"🌟 🌟 🌟 {METHOD} 全部测试完成")