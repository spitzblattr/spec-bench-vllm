import torch
from safetensors.torch import save_file

# --------------------------------------------
# 和 eagle 一样，在 vllm 中使用 Medusa 也需要将其转换为特定的模型格式，见https://github.com/vllm-project/vllm/issues/6777
input_path = "/PATH/TO/YOUR/medusa-vicuna-7b-v1.3/medusa_lm_head.pt" # download：https://huggingface.co/FasterDecoding/medusa-vicuna-7b-v1.3
output_path = "/PATH/TO/YOUR/vllm-medusa-vicuna-7b-v1.3/model.safetensors"
# --------------------------------------------
state_dict = torch.load(input_path, map_location="cpu")
new_state_dict = {}

for key, value in state_dict.items():
    # 匹配 "a.b.linear.weight/bias" 格式
    if ".linear.weight" in key or ".linear.bias" in key:
        parts = key.split(".")
        a, b = parts[0], parts[1]
        suffix = parts[-1]  # weight 或 bias
        new_key = f"blocks.{a}.layers.{b}.{suffix}"
    # 匹配 "a.b.weight" 格式
    elif ".weight" in key and "linear" not in key:
        parts = key.split(".")
        a = parts[0]
        suffix = parts[-1]  # weight
        new_key = f"lm_heads.{a}.{suffix}"
    else:
        new_key = key

    new_state_dict[new_key] = value

save_file(new_state_dict, output_path)
print(f"模型权重已保存至 {output_path}")