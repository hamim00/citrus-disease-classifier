# app.py
import os, io
import streamlit as st
import torch
from PIL import Image

# --- local modules ---
from models.registry import MODEL_CATALOG, build_model, load_class_names
from utils.transforms import build_transforms, IMAGENET_MEAN, IMAGENET_STD
from utils.inference import predict_proba, topk_from_proba
from xai.cam_utils import run_cam
from xai.lime_utils import lime_explain

# ---------------- UI SETUP ----------------
st.set_page_config(page_title="Carambola Classifier + XAI", layout="wide")
st.title("Carambola Classifier with Model Picker & XAI")

# ---------------- GLOBALS -----------------
IMG_SIZE = 224  # change if you trained with a different size

# ---------------- HELPERS -----------------
def _find_ckpt(mconf):
    """
    Resolve checkpoint path in a user-friendly way:
    1) Use provided path if it exists,
    2) Otherwise try common patterns like weights/{arch}_best.pt and variants,
    3) Return None if nothing found.
    """
    ckpt = mconf.get("ckpt")
    if ckpt and os.path.exists(ckpt):
        return ckpt

    arch = (mconf.get("arch") or "").lower()
    candidates = [
        f"weights/{arch}_best.pt",
        f"weights/{arch}.pt",
    ]
    # Graceful aliasing for your names
    if arch == "s_cnn":
        candidates.extend(["weights/scnn_best.pt", "weights/scnn.pt"])
    if arch == "custom_cnn":
        candidates.extend(["weights/custom_cnn_best.pt", "weights/custom_cnn.pt"])
    if arch == "resnet18":
        candidates.extend([
            "weights/resnet18_best.pt",
            "weights/resnet18_finetuned.pt",
            "weights/resnet18.pt",
        ])

    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def _filter_catalog_for_ui(catalog):
    """
    Show only models that are usable:
      - trained checkpoints we can find, OR
      - pretrained=True models with no checkpoint (ImageNet baselines).
    """
    usable = []
    for m in catalog:
        mc = dict(m)
        mc["ckpt_resolved"] = _find_ckpt(m)
        if mc["ckpt_resolved"] or (m.get("ckpt") is None and m.get("pretrained", False)):
            usable.append(mc)
    return usable

def _list_samples():
    root = "assets/samples"
    if not os.path.exists(root):
        return []
    files = [os.path.join("assets", "samples", f)
             for f in os.listdir(root)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    files.sort()
    return files

# ---------------- LOAD CLASS NAMES ----------------
try:
    CLASS_NAMES = load_class_names("weights/class_names.json")
except Exception as e:
    st.error(f"Could not load class names from weights/class_names.json — {e}")
    st.stop()

NUM_CLASSES = len(CLASS_NAMES)

# ---------------- SIDEBAR: MODEL PICKER ----------------
catalog = _filter_catalog_for_ui(MODEL_CATALOG)
if not catalog:
    st.error("No usable models found. Put your trained weights in the **weights/** folder "
             "(e.g., `custom_cnn_best.pt`, `s_cnn_best.pt`, `resnet18_best.pt`) or pick a "
             "pretrained baseline in `models/registry.py`.")
    st.stop()

model_names = [m["name"] for m in catalog]
choice = st.sidebar.selectbox("Choose a model", model_names)
mconf = next(m for m in catalog if m["name"] == choice)

st.sidebar.caption("Selected model")
st.sidebar.write(f"- Arch: **{mconf['arch']}**")
st.sidebar.write(f"- Pretrained: **{mconf.get('pretrained', False)}**")
st.sidebar.write(f"- Input size: **{IMG_SIZE}×{IMG_SIZE}**")
st.sidebar.write(f"- #Classes: **{NUM_CLASSES}**")
ck_txt = mconf["ckpt_resolved"] or ("— (ImageNet weights)" if mconf.get("pretrained") else "—")
st.sidebar.write(f"- Checkpoint: **{ck_txt}**")

# ---------------- BUILD & LOAD MODEL ----------------
model, target_layer = build_model(
    mconf["arch"],
    NUM_CLASSES,
    pretrained=mconf.get("pretrained", True)
)

if mconf["ckpt_resolved"]:
    try:
        sd = torch.load(mconf["ckpt_resolved"], map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=False)
    except Exception as e:
        st.warning(f"Loaded base weights; checkpoint load had a non-fatal issue: {e}")
else:
    if not mconf.get("pretrained", False):
        st.warning("No checkpoint found and model isn't marked pretrained; using random init.")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ---------------- 1) IMAGE INPUT ----------------
st.subheader("1) Upload an image or use a sample")
c1, c2 = st.columns([2, 1])

uploaded = c1.file_uploader("Upload image (JPG/PNG)", type=["jpg", "jpeg", "png"])
samples = _list_samples()
sample_choice = c2.selectbox("Samples", ["None"] + samples)

pil_in = None
if uploaded is not None:
    pil_in = Image.open(uploaded).convert("RGB")
elif sample_choice != "None" and os.path.exists(sample_choice):
    pil_in = Image.open(sample_choice).convert("RGB")

if pil_in is None:
    st.info("Upload an image or pick a sample to continue.")
    st.stop()

c1.image(pil_in, caption="Input", use_container_width=True)

# ---------------- 2) PREDICTION ----------------
st.subheader("2) Predictions")
tfm = build_transforms(IMG_SIZE)
x = tfm(pil_in).unsqueeze(0).to(device)
proba = predict_proba(model, x)
top3 = topk_from_proba(proba, CLASS_NAMES, k=3)
for lbl, p in top3:
    st.write(f"- **{lbl}**: {p:.4f}")
pred_idx = int(torch.argmax(proba, dim=1).item())

# ---------------- 3) XAI VISUALIZATIONS ----------------
st.subheader("3) XAI Visualizations (same predicted class across methods)")
cols = st.columns(5)
imgs_to_zip = []

with st.spinner("Generating CAMs & LIME…"):
    # CAM family
    for name, col in zip(["gradcam", "gradcam++", "eigencam", "ablationcam"], cols[:4]):
        vis = run_cam(name, model, target_layer, x, target_category=pred_idx)
        col.image(vis, caption=name.upper(), use_container_width=True)
        bio = io.BytesIO(); vis.save(bio, format="PNG")
        imgs_to_zip.append((f"{name}.png", bio.getvalue()))

    # LIME: needs a batch predict function on uint8 HxWx3 images
    def lime_batch_predict(batch_np):
        import numpy as np
        import torch
        from torchvision import transforms

        t = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        imgs = []
        for arr in batch_np:
            pil = Image.fromarray(arr.astype("uint8"))
            imgs.append(t(pil))
        xx = torch.stack(imgs, dim=0).to(device)
        with torch.no_grad():
            pp = predict_proba(model, xx)
        return pp.cpu().numpy()

    lime_img, lime_label = lime_explain(lime_batch_predict, pil_in, img_size=IMG_SIZE)
    cols[4].image(lime_img, caption=f"LIME (label={lime_label})", use_container_width=True)
    bio = io.BytesIO(); lime_img.save(bio, format="PNG")
    imgs_to_zip.append(("lime.png", bio.getvalue()))

# ---------------- 4) EXPORT ----------------
st.subheader("4) Download results")
if imgs_to_zip:
    import zipfile
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in imgs_to_zip:
            zf.writestr(name, data)
    st.download_button("Download ZIP",
                       data=bio.getvalue(),
                       file_name="xai_outputs.zip",
                       mime="application/zip")
