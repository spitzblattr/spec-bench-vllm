datasets_parent_dir = "/PATH/TO/YOUR/datasets/"
online_benchmark_script_path = "/PATH/TO/YOUR/[Round-1]stress_test_all_methods/auto_test_code/[Round-1]benchmark_serving.py"
task_dataset_map = {
    "ShareGPT": datasets_parent_dir+"sharegpt_processed_1024.json",
    "SpecBench_conversation": datasets_parent_dir+"spec-bench-conversation.jsonl",
    "SpecBench_math_reasoning": datasets_parent_dir+"spec-bench-math-reasoning.jsonl",
    "SpecBench_qa": datasets_parent_dir+"spec-bench-qa.jsonl",
    "SpecBench_rag": datasets_parent_dir+"spec-bench-rag.jsonl",
    "SpecBench_summarization": datasets_parent_dir+"spec-bench-summarization.jsonl",
    "SpecBench_translation": datasets_parent_dir+"spec-bench-translation.jsonl",
    "SpecBench_All": datasets_parent_dir+"spec-bench-all.jsonl",
}
python_exe_path = "/PATH/TO/YOUR/envs/bin/python"