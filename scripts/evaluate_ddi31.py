# MAAI-SkinDx | DDI-31 Benchmark Evaluation
# Fixed: calls Agent 2 for each image to provide
# the 6916-dim features required by the combined model.

import os, sys, numpy as np, pandas as pd
import torch, torch.nn as nn
import warnings; warnings.filterwarnings("ignore")
import io, contextlib

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from torchvision import transforms, models
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DDI_CSV    = os.path.join(PROJECT_ROOT, "data", "processed",
             "agent3_ddi31_test.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "agent3",
             "agent3_resnet50_final.pth")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "agent3_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE           = 224
MALIGNANT_CLASSES  = {0, 2, 3}
MELANOMA_THRESHOLD = 0.25
AGENT2_DIM         = 6916

if not os.path.exists(DDI_CSV):
    print(f"ERROR: {DDI_CSV} not found"); sys.exit(1)
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: {MODEL_PATH} not found"); sys.exit(1)

print(f"\n{'='*65}")
print(f"  MAAI-SkinDx | DDI-31 Benchmark Evaluation")
print(f"{'='*65}")
print(f"  Device   : {DEVICE}")
print(f"  Baseline : SkinGPT-X 35.8%")
print(f"  Note     : Agent 2 called for each image (correct eval)")
print(f"{'='*65}\n")

# ── Load Agent 2 ──────────────────────────────────────────────────
print("-- Loading Agent 2 --")
from agents.agent2_feature_extraction import FeatureExtractionAgent
agent2 = FeatureExtractionAgent()
print("  Agent 2 ready\n")

# ── Load Agent 3 model ────────────────────────────────────────────
print("-- Loading Agent 3 Model --")
ckpt         = torch.load(MODEL_PATH, map_location=DEVICE,
                          weights_only=False)
use_agent2   = ckpt.get("use_agent2", False)
combined_dim = ckpt.get("combined_dim", 2048)
num_classes  = ckpt.get("num_classes", 9)
print(f"  Architecture : {'COMBINED' if use_agent2 else 'BACKBONE'} {combined_dim}-dim")
print(f"  Classes      : {num_classes}")
print(f"  Use Agent2   : {use_agent2}")

class CombinedModel(nn.Module):
    def __init__(self, combined_dim, num_classes, use_agent2):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.conv1=resnet.conv1; self.bn1=resnet.bn1
        self.relu=resnet.relu; self.maxpool=resnet.maxpool
        self.layer1=resnet.layer1; self.layer2=resnet.layer2
        self.layer3=resnet.layer3; self.layer4=resnet.layer4
        self.avgpool=resnet.avgpool
        self.use_agent2=use_agent2
        self.classifier=nn.Sequential(
            nn.Dropout(0.4), nn.Linear(combined_dim, 512),
            nn.ReLU(inplace=True), nn.BatchNorm1d(512),
            nn.Dropout(0.3), nn.Linear(512, num_classes))

    def forward(self, images, a2=None):
        x=self.conv1(images); x=self.bn1(x)
        x=self.relu(x); x=self.maxpool(x)
        x=self.layer1(x); x=self.layer2(x)
        x=self.layer3(x); x=self.layer4(x)
        x=self.avgpool(x); b=torch.flatten(x, 1)
        if self.use_agent2 and a2 is not None:
            c = torch.cat([b, a2], dim=1)
        else:
            c = b
        return self.classifier(c), b

model = CombinedModel(combined_dim, num_classes, use_agent2)
model.load_state_dict(ckpt["state_dict"])
model.eval(); model = model.to(DEVICE)
print("  Loaded successfully\n")

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

# ── Load DDI-31 ───────────────────────────────────────────────────
ddi_df = pd.read_csv(DDI_CSV)
ddi_df["exists"] = ddi_df["image_path"].apply(os.path.exists)
ddi_df = ddi_df[ddi_df["exists"]].copy().reset_index(drop=True)

# Fix malignant column — handle both bool and string
ddi_df["true_malignant"] = ddi_df["malignant"].apply(
    lambda x: str(x).strip().lower() == "true")

mal_count = ddi_df["true_malignant"].sum()
ben_count = (~ddi_df["true_malignant"]).sum()
print(f"  DDI-31 images : {len(ddi_df)}")
print(f"  Malignant     : {mal_count}")
print(f"  Benign        : {ben_count}")
print(f"  Skin tones    : {sorted(ddi_df['skin_tone'].unique())}\n")

# ── Run inference with Agent 2 features ───────────────────────────
print("-- Running Inference (Agent 2 + Agent 3) --")
print(f"  Processing {len(ddi_df)} images...")
print(f"  Each image: Agent 2 extraction + Agent 3 classification")
print(f"  Estimated time: ~{len(ddi_df)*0.8/60:.0f} minutes\n")

all_probs    = []
all_pred_cls = []
a2_failed    = 0
inf_failed   = 0

model.eval()
for i, row in ddi_df.iterrows():
    img_path = row["image_path"]

    # Step 1: Get Agent 2 features (suppress verbose output)
    a2_tensor = None
    if use_agent2:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                a2_result = agent2.run(img_path)
            if (a2_result.get("status") == "PASS" and
                    a2_result.get("feature_vector") is not None):
                a2_np = np.asarray(
                    a2_result["feature_vector"], dtype=np.float32)
                if a2_np.shape[0] == AGENT2_DIM:
                    a2_tensor = torch.from_numpy(a2_np)\
                        .unsqueeze(0).to(DEVICE)
                else:
                    a2_failed += 1
            else:
                a2_failed += 1
        except Exception:
            a2_failed += 1

    # Step 2: Run Agent 3
    try:
        img    = PILImage.open(img_path).convert("RGB")
        tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out, _ = model(tensor, a2_tensor)
            probs  = torch.softmax(out, dim=1).squeeze().cpu().numpy()
        pred = int(np.argmax(probs))
        all_probs.append(probs)
        all_pred_cls.append(pred)
    except Exception as e:
        all_probs.append(np.ones(num_classes)/num_classes)
        all_pred_cls.append(1)
        inf_failed += 1

    if (i+1) % 50 == 0:
        print(f"  [{i+1:3d}/{len(ddi_df)}] "
              f"A2 failed: {a2_failed} | Inf failed: {inf_failed}")

print(f"\n  Done.")
print(f"  Agent2 failures  : {a2_failed}")
print(f"  Inference failures: {inf_failed}\n")

# ── Compute predictions ───────────────────────────────────────────
ddi_df["pred_class"]     = all_pred_cls
ddi_df["melanoma_prob"]  = [float(p[0]) for p in all_probs]
ddi_df["pred_malignant"] = [c in MALIGNANT_CLASSES
                             for c in all_pred_cls]
ddi_df["pred_mal_safe"]  = (
    ddi_df["pred_malignant"] |
    (ddi_df["melanoma_prob"] >= MELANOMA_THRESHOLD))

y_true      = ddi_df["true_malignant"].values
y_pred      = ddi_df["pred_malignant"].values
y_pred_safe = ddi_df["pred_mal_safe"].values

# ── Binary evaluation ─────────────────────────────────────────────
acc_bin  = accuracy_score(y_true, y_pred)
f1_bin   = f1_score(y_true, y_pred, zero_division=0)
prec_bin = precision_score(y_true, y_pred, zero_division=0)
rec_bin  = recall_score(y_true, y_pred, zero_division=0)

acc_safe = accuracy_score(y_true, y_pred_safe)
f1_safe  = f1_score(y_true, y_pred_safe, zero_division=0)
rec_safe = recall_score(y_true, y_pred_safe, zero_division=0)

print(f"{'='*65}")
print(f"  BINARY MALIGNANT/BENIGN RESULTS")
print(f"  SkinGPT-X baseline: 35.8%")
print(f"{'='*65}")
print(f"\n  Without melanoma safety threshold:")
print(f"    Accuracy  : {acc_bin*100:.2f}%  "
      f"{'BEATS SkinGPT-X' if acc_bin>0.358 else 'Below baseline'}")
print(f"    F1 Score  : {f1_bin*100:.2f}%")
print(f"    Precision : {prec_bin*100:.2f}%")
print(f"    Recall    : {rec_bin*100:.2f}%")
print(f"\n  With melanoma threshold ({MELANOMA_THRESHOLD}):")
print(f"    Accuracy  : {acc_safe*100:.2f}%  "
      f"{'BEATS SkinGPT-X' if acc_safe>0.358 else 'Below baseline'}")
print(f"    F1 Score  : {f1_safe*100:.2f}%")
print(f"    Recall    : {rec_safe*100:.2f}%")

cm = confusion_matrix(y_true, y_pred)
print(f"\n  Confusion Matrix:")
print(f"              Pred Benign  Pred Malignant")
print(f"  True Benign   {cm[0,0]:6d}        {cm[0,1]:6d}")
print(f"  True Malign   {cm[1,0]:6d}        {cm[1,1]:6d}")

# ── Fairness ──────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  FAIRNESS ACROSS SKIN TONE GROUPS")
print(f"{'='*65}")
skin_tone_groups = {
    "FST I-II (Light)"    : ["12","13","14","15","16"],
    "FST III-IV (Medium)" : ["34","35","36"],
    "FST V-VI (Dark)"     : ["56","65","66"],
}
fairness = {}
print(f"\n  {'Group':<22} {'N':>4} {'Accuracy':>10} "
      f"{'Mal Recall':>11} {'Specificity':>12}")
print(f"  {'─'*22} {'─'*4} {'─'*10} {'─'*11} {'─'*12}")

for grp, tones in skin_tone_groups.items():
    mask = ddi_df["skin_tone"].astype(str).isin(tones)
    g    = ddi_df[mask]
    if len(g) == 0: continue
    gt   = g["true_malignant"].values
    gp   = g["pred_malignant"].values
    gacc = accuracy_score(gt, gp)
    grec = recall_score(gt, gp, zero_division=0)
    bm   = ~gt
    gspec= (gp[bm]==False).mean() if bm.sum()>0 else 0.0
    fairness[grp] = {"n":len(g),"acc":gacc,"rec":grec,"spec":gspec}
    print(f"  {grp:<22} {len(g):>4} "
          f"{gacc*100:>9.2f}% "
          f"{grec*100:>10.2f}% "
          f"{gspec*100:>11.2f}%")

if fairness:
    accs = [v["acc"] for v in fairness.values()]
    gap  = (max(accs)-min(accs))*100
    print(f"\n  Max accuracy gap : {gap:.2f}%  "
          f"{'ACCEPTABLE' if gap<=10 else 'FAIRNESS CONCERN'}")

# ── Plot ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    f"MAAI-SkinDx | DDI-31 Benchmark\n"
    f"SkinGPT-X: 35.8% | MAAI-SkinDx: {acc_bin*100:.2f}%",
    fontsize=13, fontweight="bold")

ax = axes[0]
methods = ["SkinGPT-X\n(Baseline)", "MAAI-SkinDx\n(No threshold)",
           "MAAI-SkinDx\n(With threshold)"]
vals    = [35.8, acc_bin*100, acc_safe*100]
colors  = ["#FF7043","#2196F3","#4CAF50"]
bars    = ax.bar(range(3), vals, color=colors,
                 edgecolor="black", linewidth=0.8, width=0.5)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.3, f"{val:.1f}%",
            ha="center", va="bottom",
            fontweight="bold", fontsize=11)
ax.axhline(y=35.8, color="red", linestyle="--",
           linewidth=1.5, alpha=0.7, label="SkinGPT-X 35.8%")
ax.set_xticks(range(3)); ax.set_xticklabels(methods, fontsize=10)
ax.set_ylabel("Accuracy (%)"); ax.set_ylim([0, min(100,max(vals)+12)])
ax.set_title("Binary Classification Accuracy", fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

ax = axes[1]
if fairness:
    gn = list(fairness.keys())
    ga = [fairness[g]["acc"]*100 for g in gn]
    gr = [fairness[g]["rec"]*100 for g in gn]
    x  = np.arange(len(gn)); w=0.35
    b1 = ax.bar(x-w/2, ga, w, label="Accuracy",
                color="#2196F3", edgecolor="black", linewidth=0.8)
    b2 = ax.bar(x+w/2, gr, w, label="Malignant Recall",
                color="#FF9800", edgecolor="black", linewidth=0.8)
    for bar, val in zip(list(b1)+list(b2), ga+gr):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.3, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace(" (","\n(") for g in gn], fontsize=9)
    ax.set_ylabel("Score (%)"); ax.set_ylim([0,110])
    ax.set_title("Fairness Across Skin Tone Groups",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "ddi31_evaluation.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()

# ── Save results ──────────────────────────────────────────────────
results_path = os.path.join(OUTPUT_DIR, "ddi31_results.txt")
with open(results_path, "w") as f:
    f.write("MAAI-SkinDx | DDI-31 Benchmark Results\n")
    f.write("="*50+"\n\n")
    f.write(f"Architecture      : COMBINED {combined_dim}-dim\n")
    f.write(f"SkinGPT-X baseline: 35.8%\n\n")
    f.write(f"MAAI-SkinDx (no threshold):\n")
    f.write(f"  Accuracy  : {acc_bin*100:.2f}%\n")
    f.write(f"  F1 Score  : {f1_bin*100:.2f}%\n")
    f.write(f"  Recall    : {rec_bin*100:.2f}%\n")
    f.write(f"  Precision : {prec_bin*100:.2f}%\n\n")
    f.write(f"With threshold ({MELANOMA_THRESHOLD}):\n")
    f.write(f"  Accuracy  : {acc_safe*100:.2f}%\n")
    f.write(f"  Recall    : {rec_safe*100:.2f}%\n\n")
    if fairness:
        f.write("Fairness:\n")
        for g,r in fairness.items():
            f.write(f"  {g}: Acc={r['acc']*100:.2f}% "
                    f"Recall={r['rec']*100:.2f}%\n")

# ── Summary ───────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  FINAL SUMMARY")
print(f"{'='*65}")
print(f"  Architecture              : COMBINED {combined_dim}-dim")
print(f"  SkinGPT-X baseline        : 35.8%")
print(f"  MAAI-SkinDx (no thresh)   : {acc_bin*100:.2f}%  "
      f"{'BEATS BASELINE' if acc_bin>0.358 else 'Below baseline'}")
print(f"  MAAI-SkinDx (with thresh) : {acc_safe*100:.2f}%  "
      f"{'BEATS BASELINE' if acc_safe*100>35.8 else 'Below baseline'}")
print(f"  Improvement               : +{acc_bin*100-35.8:.2f} pts")
print(f"  Malignant recall          : {rec_bin*100:.2f}%")
print(f"  Malignant F1              : {f1_bin*100:.2f}%")
print(f"  Plot saved                : {plot_path}")
print(f"  Results saved             : {results_path}")
print(f"{'='*65}\n")