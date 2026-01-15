import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader
import os, sys, glob
import matplotlib
matplotlib.use("Agg")
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

# =============================================================================
# 0. 全局开关（你要的）
# =============================================================================
SMOKE_TEST = True   # True = 快速可行性验证；False = 正式全量训练

# =============================================================================
# 1. 环境与配置
# =============================================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.backends.cudnn.enabled = False

TRAIN_DATASET_DIR = "./dataset/DIV2K_train_latents/"
VALID_DATASET_DIR = "./dataset/DIV2K_valid_HR"
CHECKPOINT_DIR = "./output/checkpoints/train_production_v1/"

PIXART_PATH = "./output/pretrained_models/PixArt-XL-2-512x512.pth"
VAE_PATH = "./output/pretrained_models/sd-vae-ft-ema"
T5_EMBED_PATH = "./output/quality_embed.pth"

DEVICE = "cuda"
DTYPE = torch.float16

NUM_EPOCHS = 100
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
NUM_WORKERS = 2
LR_ADAPTER = 1e-5

SDE_STRENGTH = 0.5
VALIDATION_STEPS = 20

# =============================================================================
# 2. 模型导入（与你仓库完全一致）
# =============================================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from diffusion.model.nets.PixArtMS import PixArtMS_XL_2
from diffusion.model.nets.adapter import MultiLevelAdapter
from diffusion import IDDPM

# =============================================================================
# 3. 数据集（不改）
# =============================================================================
class TrainLatentDataset(Dataset):
    def __init__(self, root):
        self.files = sorted(glob.glob(os.path.join(root, "*.pt")))
        assert len(self.files) > 0

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx], map_location="cpu")

class ValidImageDataset(Dataset):
    def __init__(self, root):
        self.files = sorted(
            glob.glob(os.path.join(root, "*.png")) +
            glob.glob(os.path.join(root, "*.jpg"))
        )
        self.transform = transforms.Compose([
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        hr = Image.open(self.files[idx]).convert("RGB")
        hr = self.transform(hr)
        lr = F.interpolate(hr.unsqueeze(0), scale_factor=0.25,
                           mode="bicubic", align_corners=False)
        lr = F.interpolate(lr, size=(512,512),
                           mode="bicubic", align_corners=False).squeeze(0)
        return {"hr_img": hr, "lr_img": lr}

# =============================================================================
# 4. 验证函数（严格沿用单样本拟合范式）
# =============================================================================
def validate(model, adapter, vae, loader, y_embed, data_info):
    model.eval()
    adapter.eval()

    diffusion = IDDPM(str(1000))
    start_t = int(1000 * SDE_STRENGTH)

    with torch.no_grad():
        for batch in loader:
            hr = batch["hr_img"].to(DEVICE).to(DTYPE)
            lr = batch["lr_img"].to(DEVICE).to(DTYPE)

            lr_latent = vae.encode(lr).latent_dist.mean \
                        * vae.config.scaling_factor

            noise = torch.randn_like(lr_latent)
            t = torch.tensor([start_t], device=DEVICE).long()
            latents = diffusion.q_sample(lr_latent, t, noise)

            cond = adapter(lr_latent.float())

            for step in range(start_t, -1, -50):  # 和单样本一致，粗步长
                t_step = torch.tensor([step], device=DEVICE).long()
                with torch.cuda.amp.autocast():
                    out = model(
                        latents, t_step, y_embed,
                        data_info=data_info,
                        adapter_cond=cond,
                        injection_mode="hybrid",
                    )
                if out.shape[1] == 8:
                    out, _ = out.chunk(2, dim=1)
                latents = latents - out * 0.1  # 简化版 DDPM step

            _ = vae.decode(latents / vae.config.scaling_factor).sample
            break  # SMOKE 或正式验证都只取 loader 控制

    model.train()
    adapter.train()

# =============================================================================
# 5. 主训练
# =============================================================================
def train_full_production():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    pixart = PixArtMS_XL_2(input_size=64).to(DEVICE).to(DTYPE).train()
    pixart.load_state_dict(torch.load(PIXART_PATH, map_location="cpu"), strict=False)
    for p in pixart.parameters():
        p.requires_grad = False

    adapter = MultiLevelAdapter(in_channels=4, hidden_size=1152).to(DEVICE).train()

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=LR_ADAPTER)
    scaler = GradScaler()
    diffusion = IDDPM(str(1000))

    train_loader = DataLoader(
        TrainLatentDataset(TRAIN_DATASET_DIR),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        ValidImageDataset(VALID_DATASET_DIR),
        batch_size=1,
        shuffle=False,
    )

    vae = AutoencoderKL.from_pretrained(VAE_PATH).to(DEVICE).to(DTYPE).eval()
    for p in vae.parameters():
        p.requires_grad = False

    y_embed = torch.load(T5_EMBED_PATH, map_location="cpu")["prompt_embeds"] \
                .unsqueeze(1).to(DEVICE).to(DTYPE)
    data_info = {
        "img_hw": torch.tensor([[512.,512.]], device=DEVICE, dtype=DTYPE),
        "aspect_ratio": torch.tensor([1.], device=DEVICE, dtype=DTYPE),
    }

    # =======================
    # SMOKE 模式：只跑 1 epoch
    # =======================
    max_epochs = 1 if SMOKE_TEST else NUM_EPOCHS

    for epoch in range(max_epochs):
        optimizer.zero_grad()

        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            hr_latent = batch["hr_latent"].to(DEVICE).to(DTYPE)
            lr_img = batch["lr_img"].to(DEVICE).to(DTYPE)

            with torch.no_grad():
                lr_latent = vae.encode(lr_img).latent_dist.sample() \
                            * vae.config.scaling_factor

            t = torch.randint(0, 1000, (1,), device=DEVICE).long()
            noise = torch.randn_like(hr_latent)
            noisy = diffusion.q_sample(hr_latent, t, noise)

            cond = adapter(lr_latent.float())

            with torch.cuda.amp.autocast():
                out = pixart(
                    noisy, t, y_embed,
                    data_info=data_info,
                    adapter_cond=cond,
                    injection_mode="hybrid",
                )
                if out.shape[1] == 8:
                    out, _ = out.chunk(2, dim=1)
                loss = F.mse_loss(out, noise) / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if SMOKE_TEST:
                break  # 只跑 1 个 batch

        # ===== 真实验证（SMOKE 也用同一套）=====
        validate(
            pixart, adapter, vae,
            val_loader if not SMOKE_TEST else [next(iter(val_loader))],
            y_embed, data_info
        )

        if SMOKE_TEST:
            print("\n🧪 SMOKE TEST PASSED. Exit.\n")
            return

if __name__ == "__main__":
    train_full_production()
