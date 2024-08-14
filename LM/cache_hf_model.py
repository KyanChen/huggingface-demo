import mmengine
from transformers import AutoModelForCausalLM, AutoTokenizer

model_repo = "Qwen/Qwen2-1.5B-Instruct"
save_dir = "work_dirs/model_cache"
mmengine.mkdir_or_exist(save_dir)

model = AutoModelForCausalLM.from_pretrained(
    model_repo,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_repo)

# save the model and tokenizer to local disk
model.save_pretrained(save_dir+"/Qwen2-1.5B-Instruct")
tokenizer.save_pretrained(save_dir+"/Qwen2-1.5B-Instruct")

