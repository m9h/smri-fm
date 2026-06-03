"""Modal app: parallel FOMO26 CLS002 (infarct) diffusion-classification sweep.

Each (arm, task, fold) is an independent GPU job, so the 7-arm x {full
flair+adc+dwi, dwi-only} x 5-fold rotating-CV matrix that takes ~2-4 days serial
on the local GB10 fans out across cloud GPUs and finishes in ~one run's wall time.

Reuses the EXACT pretrained checkpoints + asparagus finetune CLI validated
locally; only the orchestration moves to Modal. Weights + the two preprocessed
cohorts live on a Modal Volume (uploaded once via scripts/modal_upload.sh); the
small repo code (asparagus + bridges) ships in the image.

Run:
  modal run scripts/modal_cls002_sweep.py                      # full 70-run sweep
  modal run scripts/modal_cls002_sweep.py --arms smri_siam --tasks CLS002_FOMO26_Infarct --folds 0   # one run
Results (per-run predictions + an aggregated leaderboard json) are written back
to the Volume under /fomo26/results and also printed.
"""
from __future__ import annotations

import json
import os
import subprocess

import modal

REPO_LOCAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = modal.App("fomo26-cls002")
vol = modal.Volume.from_name("fomo26-cls002", create_if_missing=True)

image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:26.04-py3")
    .pip_install(
        "gardening_tools", "monai==1.5.2", "lightning==2.5.0", "torchmetrics",
        "hydra-core", "omegaconf", "python-dotenv", "wandb", "nnunetv2",
        "nibabel", "scikit-image", "scikit-learn", "pandas", "scipy", "einops",
    )
    .add_local_dir(f"{REPO_LOCAL}/third_party/asparagus",
                   "/workspace/smri-fm/third_party/asparagus")
    .add_local_dir(f"{REPO_LOCAL}/third_party/fomo25_mmunetvae",
                   "/workspace/smri-fm/third_party/fomo25_mmunetvae")
    .add_local_dir(f"{REPO_LOCAL}/src", "/workspace/smri-fm/src")
)

ALL_ARMS = ["smri_siam", "smri_fomo60k", "smri_anatcl", "smri_simclr3d",
            "smri_triad", "smri_brainiac", "smri_mmunetvae"]
ALL_TASKS = ["CLS002_FOMO26_Infarct", "CLS002_FOMO26_Infarct_DWI"]
ALL_FOLDS = [0, 1, 2, 3, 4]

# Per-arm checkpoint conversion: (module, fn, source-path-on-volume).
# SIAM is special (convert_checkpoint('smri_siam', dir/fold_0/ckpt) + needs SIAM_MODEL_DIR).
CONVERT = {
    "smri_fomo60k":   ("asparagus_bridge.models_smri_fomo60k",   "convert_fomo60k_checkpoint",   "/fomo26/weights/fomo60k_pkoutsouvelis/combined_regular-step=200000.ckpt"),
    "smri_brainiac":  ("asparagus_bridge.models_smri_brainiac",  "convert_brainiac_checkpoint",  "/fomo26/weights/brainiac/BrainIAC.ckpt"),
    "smri_triad":     ("asparagus_bridge.models_smri_triad",     "convert_triad_checkpoint",     "/fomo26/weights/triad/Triad-SwinB-MAE.pth"),
    "smri_anatcl":    ("asparagus_bridge.models_smri_anatcl",    "convert_anatcl_checkpoint",    "/fomo26/weights/anatcl/anatcl_global_fold0.pth"),
    "smri_mmunetvae": ("asparagus_bridge.models_smri_mmunetvae", "convert_mmunetvae_checkpoint", "/fomo26/weights/mmunetvae/fomo25_mmunetvae_pretrained.ckpt"),
    "smri_simclr3d":  ("asparagus_bridge.models_smri_simclr3d",  "convert_simclr3d_checkpoint",  "/fomo26/weights/simclr3d/simclr_3d_brain_foundation.tar"),
}
SIAM_MODEL_DIR = "/fomo26/siam_params/pred_DS108_LcsfP_Ano"


def _setenv() -> dict:
    repo = "/workspace/smri-fm"
    env = dict(os.environ)
    env.update(
        ASPARAGUS_CONFIGS=f"{repo}/third_party/asparagus/configs",
        ASPARAGUS_FINETUNE_CONFIGS=f"{repo}/src/asparagus_bridge/configs",
        ASPARAGUS_DATA="/fomo26/processed",
        ASPARAGUS_RAW_LABELS="/fomo26/raw_labels",
        ASPARAGUS_MODELS="/fomo26/models",
        ASPARAGUS_RESULTS="/fomo26/results",
        SIAM_MODEL_DIR=SIAM_MODEL_DIR,
        PYTHONPATH=f"{repo}/third_party/asparagus:{repo}/src",
        WANDB_MODE="offline", WANDB_DISABLED="true", HYDRA_FULL_ERROR="1",
    )
    return env


@app.function(image=image, gpu="A100", volumes={"/fomo26": vol},
              timeout=60 * 60, retries=1)
def run_one(arm: str, task: str, fold: int, epochs: int = 25) -> dict:
    import sys
    env = _setenv()
    sys.path[:0] = ["/workspace/smri-fm/third_party/asparagus", "/workspace/smri-fm/src"]
    for k in ("ASPARAGUS_MODELS", "ASPARAGUS_RESULTS", "ASPARAGUS_RAW_LABELS"):
        os.makedirs(env[k], exist_ok=True)

    ckpt = f"/tmp/{arm}_cls002.ckpt"
    try:
        if arm == "smri_siam":
            from asparagus_bridge.checkpoint import convert_checkpoint
            convert_checkpoint("smri_siam", f"{SIAM_MODEL_DIR}/fold_0/checkpoint_final.pth", ckpt)
        else:
            mod, fn, src = CONVERT[arm]
            import importlib
            getattr(importlib.import_module(mod), fn)(src, ckpt)
    except Exception as e:
        import traceback
        return {"arm": arm, "task": task, "fold": fold, "ok": False,
                "rundir": None, "predictions": None,
                "stderr_tail": f"checkpoint conversion failed:\n{traceback.format_exc()}"[-2000:]}

    suffix = task[len("CLS002_FOMO26_Infarct"):] or "_FULL"
    rundir = f"/fomo26/models/cls002_{arm[len('smri_'):]}{suffix}_cv{fold}"
    cmd = [
        "python", "-m", "asparagus.pipeline.run.finetune_cls",
        f"task={task}", f"+model={arm}", f"checkpoint_path={ckpt}",
        "data.train_split=split_cv5", f"data.test_split=TEST_cv{fold}", f"data.fold={fold}",
        "training.batch_size=1", "training.load_decoder=False",
        f"training.epochs={epochs}", "training.target_size=[128,128,128]",
        "training.warmup_epochs=5", "hardware.num_workers=8",
        "logger.wandb_logging=False", f"hydra.run.dir={rundir}",
    ]
    proc = subprocess.run(cmd, env=env, cwd="/workspace/smri-fm",
                          capture_output=True, text=True)
    ok = proc.returncode == 0
    pred_path = f"{rundir}/predictions/{task}__TEST_cv{fold}__best.json"
    preds = None
    if os.path.exists(pred_path):
        preds = json.load(open(pred_path))
    vol.commit()
    return {"arm": arm, "task": task, "fold": fold, "ok": ok,
            "rundir": rundir, "predictions": preds,
            "stderr_tail": proc.stderr[-2000:] if not ok else ""}


@app.local_entrypoint()
def main(arms: str = "", tasks: str = "", folds: str = "", epochs: int = 25):
    A = arms.split(",") if arms else ALL_ARMS
    T = tasks.split(",") if tasks else ALL_TASKS
    F = [int(x) for x in folds.split(",")] if folds else ALL_FOLDS
    jobs = [(a, t, f, epochs) for a in A for t in T for f in F]
    print(f">>> launching {len(jobs)} CLS002 runs on Modal "
          f"(arms={len(A)} tasks={len(T)} folds={len(F)})")
    results = list(run_one.starmap(jobs))
    n_ok = sum(r["ok"] for r in results)
    print(f">>> done: {n_ok}/{len(results)} OK")
    for r in results:
        flag = "OK " if r["ok"] else "FAIL"
        print(f"  {flag} {r['arm']:16s} {r['task']:28s} fold{r['fold']}")
        if not r["ok"]:
            print("       " + r["stderr_tail"].replace("\n", "\n       ")[-600:])
    # stash raw results to the volume for downstream aggregation
    import datetime
    out = f"/tmp/cls002_modal_results.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f">>> wrote {out} (also copy under volume via a follow-up if needed)")
