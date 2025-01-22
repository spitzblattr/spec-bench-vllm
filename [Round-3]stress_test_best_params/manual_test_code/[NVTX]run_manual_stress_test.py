#import subprocess
import time
import requests
import os
import os
import pycuda.driver as cuda
pycuda_init_done = False  # global Flag
from multiprocessing import Process

# ------------------------------------------------------------
# max-concurr = inf  
# speculative_max_model_len: Optional[int]. 注意这个值会覆盖小草稿模型的【context length】(也就是 prompt+output 的总长度)
# speculative_disable_by_batch_size: Optional[int]
# ------------------------------------------------------------
model_path = "/PATH/TO/YOUR/Qwen2.5-14B-Instruct"
speculative_model_path = "/PATH/TO/YOUR/Qwen2.5-0.5B-Instruct"
python_executable = "/PATH/TO/YOUR/envs/bin/python"
# ------------------------------------------------------------

def run_vllm_server(port):
    vllm_server_command = None
    if port == "7777":
        # vanilla
        vllm_server_command = [
            python_executable, "-m", "vllm.entrypoints.openai.api_server",
            "--host", "0.0.0.0",
            "--port", str(port),
            "--model", model_path,
            "--dtype", "auto",
            #"--quantization", "awq_marlin",
            "--tensor-parallel-size", "1",                          
            "--gpu-memory-utilization", "0.9",
            "--disable-log-requests",
            #"--num-scheduler-steps", "4"   # 可以稍稍提升吞吐量，但不多，因为本来两个token之间的空白间隔就很小
        ]
        def _worker():
            os.environ["CUDA_VISIBLE_DEVICES"] = "4"  
            # 使用 PyCUDA 需要先调用 cuda.init() (只需要调用一次即可)
            global pycuda_init_done
            if not pycuda_init_done:
                cuda.init()
                pycuda_init_done = True     
            dev = cuda.Device(0)
            ctx = dev.make_context()
            ctx.pop()
            ctx.detach()
            # 清理完毕后，执行外部命令
            os.execvp(python_executable, vllm_server_command)
        p = Process(target=_worker, daemon=False)
        p.start()
        return p
    
    else:
        # spec decoding
        vllm_server_command = [
            python_executable, "-m", "vllm.entrypoints.openai.api_server",
            "--host", "0.0.0.0",
            "--port", str(port),
            "--model", model_path,
            "--dtype", "auto",
            #"--quantization", "awq_marlin",
            "--tensor-parallel-size", "1",                          
            "--speculative-model", speculative_model_path,
            #"--speculative-model-quantization", "awq_marlin",
            "--gpu-memory-utilization", "0.9",
            "--use-v2-block-manager",
            "--num-speculative-tokens", "3",
            # ------ 🤔 DEBUG ------
            "--disable-log-requests",
            #"--enable-chunked-prefill",   
            #"--scheduler-delay-factor", "0.5",      
            #"--num-scheduler-steps", "4",            #   File "vllm/engine/arg_utils.py", line 1138, in create_engine_config, ValueError: Speculative decoding is not supported with multi-step (--num-scheduler-steps > 1)
            "--speculative-max-model-len", "1280",
            #"--speculative-disable-by-batch-size", "8"
        ]
        def _worker():
            os.environ["CUDA_VISIBLE_DEVICES"] = "5"
            global pycuda_init_done
            if not pycuda_init_done:
                cuda.init()
                pycuda_init_done = True   
            dev = cuda.Device(0)
            ctx = dev.make_context()
            ctx.pop()
            ctx.detach()
            os.execvp(python_executable, vllm_server_command)
        p = Process(target=_worker, daemon=False)
        p.start()
        return p



def run_stress_test_client(task_name, dataset_path, client_script_path, port):
    
    client_command = [
        python_executable, client_script_path,
        "--backend", "openai-chat",
        "--base-url", f"http://localhost:{port}",  
        "--endpoint", "/v1/chat/completions",        
        "--dataset-name", task_name,      
        "--dataset-path", dataset_path,
        "--model", model_path,
        "--request-rate", "inf",       
        "--disable-tqdm",     
        "--seed", "12345",
        "--num-prompts", "120",    
        "--max-concurrency", "8"
    ]

    def _worker():
        os.execvp(python_executable, client_command)
    p = Process(target=_worker, daemon=False)
    p.start()
    return p

    
if __name__ == "__main__":
    dataset_path = "/PATH/TO/YOUR/datasets/Spec-Decode-Merged/cnn_merged.jsonl"  # 每种长度各20条
    client_script_path = "/PATH/TO/YOUR/[Round-3]stress_test_best_params/auto_test_code/[Round-3]benchmark_serving.py"
    TASK = "cnn_dailymail_news"   #"cnn_dailymail_highlights","cnn_dailymail_news", "THUCNews"
    PORTS = ["7777", "7778"]
    # 并行启动两个 vllm 服务
    vllm_processes = []
    for port in PORTS:
        vllm_process = run_vllm_server(port)
        vllm_processes.append((vllm_process, port))

    # 等待服务启动
    time.sleep(200)
    for port in PORTS:
        try:
            r = requests.get(f"http://0.0.0.0:{port}/v1/models")
            if r.status_code == 200:
                print(f"============ 开启 vllm openai server, port={port} ============")
        except requests.exceptions.ConnectionError:
            raise TimeoutError("服务启动失败")

    # 压力测试
    for task in [TASK]:
        client_processes = []
        for port in PORTS:
            print(f"\n\t 🎥 开始向 port={port} 发送请求, task={task}\n")

            process = run_stress_test_client(
                    task_name=task,
                    dataset_path=dataset_path,
                    client_script_path=client_script_path,
                    port=port
                )
            client_processes.append(process)
        
        # 等待所有client子进程完成
        for process in client_processes:
            process.join()

    # 关闭所有 vllm 服务
    print(f"🌟 🌟 🌟 测试已完成，现在关闭所有 vllm 服务...")
    for vllm_process, port in vllm_processes:
        if vllm_process.is_alive():
            pid = vllm_process.pid  # 在关之前获取 pid
            vllm_process.terminate() 
            vllm_process.join()
            print(f"已关闭 vllm 服务: pid={pid}, port={port}")

