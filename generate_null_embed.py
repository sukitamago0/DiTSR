import torch
import os
import sys

# ================= 配置 =================
# 指向你已经存在的本地权重目录
LOCAL_T5_PATH = "output/pretrained_models/t5-v1_1-xxl" 
OUTPUT_PATH = "output/null_embed.pth"
MAX_LENGTH = 120
# =======================================

def extract_offline():
    print(f"\n🚀 [CPU模式] 正在加载本地 T5 权重...")
    print(f"   路径: {LOCAL_T5_PATH}")

    # 1. 路径检查
    if not os.path.exists(LOCAL_T5_PATH):
        print(f"❌ 错误：找不到路径 {LOCAL_T5_PATH}")
        print("   请确认你已经上传了完整的 T5 文件夹。")
        return

    try:
        from transformers import T5EncoderModel, T5Tokenizer
        
        # 强制使用 CPU，避免占用 3070 的 8G 显存
        device = "cpu"
        
        # 2. 加载 Tokenizer (本地)
        print("   -> Loading Tokenizer...")
        tokenizer = T5Tokenizer.from_pretrained(LOCAL_T5_PATH, local_files_only=True)
        
        # 3. 加载 Model (本地)
        print("   -> Loading Model (这需要一点时间读取硬盘)...")
        text_encoder = T5EncoderModel.from_pretrained(
            LOCAL_T5_PATH, 
            local_files_only=True, 
            torch_dtype=torch.float32 # CPU 用 float32 兼容性最好
        ).to(device)
        
        text_encoder.eval()
        print("   ✅ T5 加载成功！")

    except Exception as e:
        print(f"\n❌ 加载失败。错误信息: {e}")
        print("   请检查文件夹内是否包含 config.json, spiece.model, pytorch_model.bin 等所有文件。")
        return

    # 4. 提取特征
    prompts = [""] 
    print("🔄 正在提取空文本特征...")
    
    with torch.no_grad():
        text_inputs = tokenizer(
            prompts,
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        
        prompt_embeds = text_encoder(
            text_inputs.input_ids.to(device),
            attention_mask=text_inputs.attention_mask.to(device),
        )[0]

    # 5. 保存
    payload = {
        "prompt_embeds": prompt_embeds, 
        "attention_mask": text_inputs.attention_mask
    }
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(payload, OUTPUT_PATH)
    print(f"\n🎉 成功！null_embed.pth 已生成至: {OUTPUT_PATH}")
    print(f"   Shape: {prompt_embeds.shape}")

if __name__ == "__main__":
    extract_offline()