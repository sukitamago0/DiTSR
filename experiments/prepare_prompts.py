import torch
import os
import sys
# [关键] 引入 transformers 库
try:
    from transformers import T5EncoderModel, T5Tokenizer
except ImportError:
    print("❌ 缺少 transformers 库，请运行: pip install transformers sentencepiece")
    sys.exit(1)

# ================= 🔧 配置区域 =================
# [请修改这里] 你的 T5 权重所在的文件夹路径 (不要指向具体文件，指向文件夹！)
# 例如: "../output/pretrained_models/t5-v1_1-xxl"
T5_LOCAL_PATH = "../output/pretrained_models/t5-v1_1-xxl" 

# 输出路径
OUTPUT_PATH = "../output/quality_embed.pth"

# 高清提示词 (Quality Prompt)
PROMPT = "cinematic photo, highly detailed, 4k, realistic, sharp focus, high resolution"
MAX_LENGTH = 120
# ===============================================

def main():
    print(f"🚀 Starting T5 Prompt Encoding (Local Mode)...")
    print(f"📂 T5 Path: {T5_LOCAL_PATH}")
    
    # 检查路径是否存在
    if not os.path.exists(T5_LOCAL_PATH):
        print(f"❌ Error: T5 path not found: {T5_LOCAL_PATH}")
        print("   请在脚本中修改 T5_LOCAL_PATH 为你实际存放 bin/json 文件的文件夹路径。")
        return

    # 1. 强制 CPU 加载 (3070 8G 扛不住 T5-XXL)
    device = "cpu"
    print("   Using Device: CPU (To save GPU VRAM)")
    
    # 2. 加载 Tokenizer
    print("⏳ Loading Tokenizer...")
    try:
        # local_files_only=True 确保不联网，只用本地
        tokenizer = T5Tokenizer.from_pretrained(T5_LOCAL_PATH, local_files_only=True)
    except Exception as e:
        print(f"❌ Tokenizer load failed: {e}")
        print("   请检查文件夹里是否有 tokenizer.json 或 spiece.model")
        return

    # 3. 加载 T5 Model (分片权重会自动处理)
    print("⏳ Loading T5-XXL Model (读取分片权重)...")
    try:
        # low_cpu_mem_usage=True 是关键，它能优化分片加载的内存占用
        # torch_dtype=torch.float32 保证 CPU 兼容性
        model = T5EncoderModel.from_pretrained(
            T5_LOCAL_PATH, 
            local_files_only=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32 
        ).to(device).eval()
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        print("   请检查文件夹里是否有 config.json 和 pytorch_model-*.bin 文件")
        return
        
    print("✅ Model Loaded Successfully!")

    # 4. Tokenize
    print(f"🔄 Processing Prompt: '{PROMPT}'")
    text_inputs = tokenizer(
        PROMPT,
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation=True,
        return_tensors="pt"
    )
    
    text_input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)
    
    # 5. Inference
    print("🔄 Encoding (Running Forward)...")
    with torch.no_grad():
        prompt_embeds = model(
            input_ids=text_input_ids,
            attention_mask=attention_mask,
        )[0] # [1, 120, 4096]
        
    print(f"✅ Generated Embed Shape: {prompt_embeds.shape}")
    
    # 6. 保存
    save_dict = {
        "prompt_embeds": prompt_embeds.float(), 
        "attention_mask": attention_mask
    }
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(save_dict, OUTPUT_PATH)
    print(f"💾 Saved to {OUTPUT_PATH}")
    print("🎉 T5 离线处理完成！现在可以去跑训练脚本了。")

if __name__ == "__main__":
    main()