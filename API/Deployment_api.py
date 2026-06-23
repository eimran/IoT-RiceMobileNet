from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Config ────────────────────────────────────────────────────────────────────
# Replace EC2_HOST with your actual EC2 public DNS, e.g.:
#   "http://ec2-13-234-56-78.compute-1.amazonaws.com"
# If you use HTTPS (via ACM + ALB), change to "https://..."
EC2_HOST      = "http://your-ec2-public-dns-here"

# Matches save_model() in the notebook (Cell 3):
#   path = f'Model/{name}_weights.pth'
#   torch.save({'model_state_dict': ..., 'class_to_idx': ..., 'class_names': ...}, path)
# → file saved to: Model/ProposedModel_weights.pth  (relative to where you run uvicorn)
# On EC2: cd /home/ubuntu/app && uvicorn main:app --host 127.0.0.1 --port 8001
WEIGHTS_PATH  = "Model/ProposedModel_weights.pth"

NUM_CLASSES   = 6            # must match training NUM_CLASSES = 6
IMAGE_SIZE    = (224, 224)   # must match training IMAGE_SIZE  = (224, 224)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB — reject before PIL/PyTorch even touch it

# Fallback class list — used ONLY if the checkpoint has no 'class_names' key.
# Order is alphabetical (torchvision.datasets.ImageFolder sorts folder names).
# Verified against notebook Cell 2: class_names = train_dataset.classes
_FALLBACK_CLASS_NAMES = [
    "bacterial_leaf_blight",
    "brown_spot",
    "healthy",
    "leaf_blast",
    "leaf_scald",
    "narrow_brown_spot",
]

# ── Model Definition ──────────────────────────────────────────────────────────
# Copied verbatim from Cell 22 of PROPOSED_MODEL.ipynb.
#
# KEY POINT — weights=None vs IMAGENET1K_V1:
#   Training used weights=IMAGENET1K_V1 (to get pretrained backbone values).
#   At inference, weights=None is correct because load_state_dict() below
#   overwrites EVERY parameter with the saved trained values.
#   The only thing that must match training is the ARCHITECTURE (layer names
#   and shapes) — and weights=None gives exactly the same architecture.
#
# KEY POINT — apply_freeze_like_keras() omitted:
#   That function only sets requires_grad = False on certain parameters.
#   It does NOT add, remove, or rename any layers, so the state-dict keys
#   are identical with or without it.
#   At inference we never compute gradients (torch.no_grad() + model.eval()),
#   so requires_grad is irrelevant.

class ProposedModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        mbnet          = torchvision.models.mobilenet_v2(weights=None)
        self.backbone  = mbnet.features       # output: (B, 1280, 7, 7) for 224×224 input

        self.depthwise = nn.Conv2d(1280, 1280, kernel_size=3, padding=1, groups=1280)
        self.pointwise = nn.Conv2d(1280, 128,  kernel_size=1)
        self.bn1       = nn.BatchNorm2d(128)
        self.gap       = nn.AdaptiveAvgPool2d(1)
        self.bn2       = nn.BatchNorm1d(128)
        self.fc1       = nn.Linear(128, 128)
        self.dropout   = nn.Dropout(0.4)
        self.fc2       = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = F.relu(self.depthwise(x), inplace=True)
        x = F.relu(self.pointwise(x), inplace=True)
        x = self.bn1(x)
        x = self.gap(x).flatten(1)
        x = self.bn2(x)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout(x)
        return self.fc2(x)   # raw logits — softmax applied in /predict, not here


# ── Load weights ──────────────────────────────────────────────────────────────
# Notebook save_model() (Cell 3) saves exactly:
#   {
#       'model_state_dict': model.state_dict(),
#       'class_to_idx':     class_to_idx,
#       'class_names':      class_names,
#   }

def load_model():
    model = ProposedModel(num_classes=NUM_CLASSES)

    # weights_only=False is required because the checkpoint contains
    # non-tensor objects (class_names list, class_to_idx dict).
    # torch.load() default is weights_only=False — stated explicitly for clarity.
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=False)

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            f"'model_state_dict' key missing in '{WEIGHTS_PATH}'. "
            "Ensure WEIGHTS_PATH points to the file produced by save_model()."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    # eval() is critical:
    #   • Dropout is disabled  (uses identity at inference)
    #   • BatchNorm uses stored running_mean/running_var instead of batch stats
    model.eval()

    names = checkpoint.get("class_names") or _FALLBACK_CLASS_NAMES
    return model, names


try:
    MODEL, CLASS_NAMES = load_model()
except Exception as exc:
    raise RuntimeError(
        f"Could not load model weights from '{WEIGHTS_PATH}': {exc}"
    ) from exc

# Catch NUM_CLASSES mismatch at startup — better than a silent wrong prediction.
if len(CLASS_NAMES) != NUM_CLASSES:
    raise ValueError(
        f"NUM_CLASSES={NUM_CLASSES} but checkpoint has {len(CLASS_NAMES)} "
        f"class names: {CLASS_NAMES}. Fix NUM_CLASSES or check the checkpoint."
    )


# ── Preprocessing ─────────────────────────────────────────────────────────────
# Must exactly mirror VAL_TRANSFORM from Cell 2 of PROPOSED_MODEL.ipynb:
#
#   MINUSONE_NORM   = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#   VAL_TRANSFORM   = transforms.Compose([
#       transforms.Resize(IMAGE_SIZE),   # → (224, 224)
#       transforms.ToTensor(),           # [0, 255] uint8 → [0.0, 1.0] float32
#       MINUSONE_NORM,                   # [0.0, 1.0]    → [−1.0, 1.0]
#   ])
#
# This is mathematically identical to Keras Rescaling(1/127.5, offset=-1).
# ToTensor() also converts HWC→CHW and handles uint8→float32.
# RandomRotation and RandomAffine are in TRAIN_TRANSFORM only — they are NOT
# in VAL_TRANSFORM and must NOT be applied at inference.

TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

def read_image_as_tensor(data: bytes) -> torch.Tensor:
    """Raw image bytes → preprocessed float32 tensor of shape (1, 3, 224, 224)."""
    image = Image.open(BytesIO(data)).convert("RGB")  # RGBA/grayscale/palette → RGB
    tensor = TRANSFORM(image)                          # (3, 224, 224)
    return tensor.unsqueeze(0)                         # (1, 3, 224, 224)


# ── FastAPI App ───────────────────────────────────────────────────────────────
# root_path="/api" is correct ONLY when nginx strips the /api prefix before
# forwarding to uvicorn, i.e. your nginx location block looks like:
#
#   location /api/ {
#       proxy_pass http://127.0.0.1:8001/;   # <-- trailing slash strips /api
#   }
#
# If nginx passes the full path (no trailing slash), remove root_path="/api".
# Without the correct setting, /docs and the OpenAPI JSON URL will be wrong.

app = FastAPI(title="Plant Disease Classifier", version="1.0", root_path="/api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        EC2_HOST,                # nginx-served frontend on port 80
        "http://localhost",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/ping")
async def ping():
    """
    Health-check.
    curl http://<ec2-host>/api/ping
    Verifies the server is up, the model loaded, and shows which device is active.
    """
    return {
        "status":  "healthy",
        "device":  str(DEVICE),
        "classes": len(CLASS_NAMES),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Classify a rice-leaf image.

    Request : POST /predict  multipart/form-data  field="file"
    Response: {"class": "<disease_name>", "confidence": <float 0-100>}
    """

    # ── Step 1: MIME-type guard ───────────────────────────────────────────────
    # content_type is None when the client sends no Content-Type header.
    # Check for None before .startswith() to avoid AttributeError.
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image (JPEG, PNG, WEBP, etc.)."
        )

    raw = await file.read()

    # ── Step 2: Size guard ────────────────────────────────────────────────────
    # Reject before PIL/PyTorch even touch the bytes.
    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File is {len(raw) / (1024 * 1024):.1f} MB — exceeds the "
                f"{MAX_FILE_SIZE // (1024 * 1024)} MB limit."
            ),
        )

    # ── Step 3: Decode & preprocess ───────────────────────────────────────────
    try:
        img_tensor = read_image_as_tensor(raw)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode or preprocess image: {exc}"
        )

    # ── Step 4: Inference ─────────────────────────────────────────────────────
    img_tensor = img_tensor.to(DEVICE)

    with torch.no_grad():
        logits     = MODEL(img_tensor)                         # (1, NUM_CLASSES)
        probs      = torch.softmax(logits, dim=1).squeeze(0)  # (NUM_CLASSES,)
        pred_idx   = int(probs.argmax().item())
        confidence = float(probs[pred_idx].item())

    return {
        "class":      CLASS_NAMES[pred_idx],
        "confidence": round(confidence * 100, 2),  # 0.9731 → 97.31
    }


# ── Entry point ───────────────────────────────────────────────────────────────
# Development:
#   python main.py
#
# Production (bind to loopback; nginx proxies from port 80):
#   uvicorn main:app --host 127.0.0.1 --port 8001
#
# Production with multiple workers:
#   gunicorn -k uvicorn.workers.UvicornWorker main:app \
#            --bind 127.0.0.1:8001 --workers 2
#
# IMPORTANT: always bind to 127.0.0.1, NOT 0.0.0.0.
# Port 8001 must NOT be open in the EC2 Security Group — only ports 80/443
# need to be open. nginx on the same instance proxies to 127.0.0.1:8001.

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)