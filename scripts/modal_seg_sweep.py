"""Modal app: parallel ISLES22 stroke-lesion seg sweep (FM vs scratch, 5-fold).

Each (arm, fold) is an independent A100 job, so the 3-arm x 5-fold matrix that
takes ~30 h serial on the local GB10 fans out across cloud GPUs and finishes in
~one run's wall time (~1.5-2 h). Reuses the EXACT checkpoints + asparagus
finetune_seg CLI validated locally (incl. the SlidingWindowSegMixin anisotropic
pad-to-64 fix, which ships in the image via src/); only orchestration moves to
Modal. The SEG011 cohort + checkpoints + SIAM plans live on the shared
`fomo26-cls002` Volume.

Run:
  modal run scripts/modal_seg_sweep.py                                  # full 15-run sweep
  modal run scripts/modal_seg_sweep.py --arms smri_siam --folds 0       # one run
"""
from __future__ import annotations

import json
import os
import subprocess

import modal

REPO_LOCAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = modal.App("fomo26-seg-isles22")
vol = modal.Volume.from_name("fomo26-cls002", create_if_missing=True)  # shared volume (has ckpts + SIAM plans)

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

# Only the seg-decoder-capable arms: SIAM nnU-Net, mmunetvae native seg, and the
# from-scratch nnU-Net control (SIAM topology, random init). Encoder-only arms
# (fomo60k/anatcl/triad/simclr3d/brainiac) would need bolt-on decoders — separate.
ALL_ARMS = ["smri_siam", "smri_mmunetvae", "scratch_nnunet"]
TASK = "SEG011_ISLES22_IschStroke"
ALL_FOLDS = [0, 1, 2, 3, 4]
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
              timeout=60 * 60 * 4, retries=1)
def run_one(arm: str, fold: int, debug: int = 0) -> dict:
    import sys
    env = _setenv()
    sys.path[:0] = ["/workspace/smri-fm/third_party/asparagus", "/workspace/smri-fm/src"]
    for k in ("ASPARAGUS_MODELS", "ASPARAGUS_RESULTS", "ASPARAGUS_RAW_LABELS"):
        os.makedirs(env[k], exist_ok=True)

    # checkpoint conversion (scratch = random init, no ckpt)
    ckpt = None
    try:
        if arm == "smri_siam":
            from asparagus_bridge.checkpoint import convert_checkpoint
            ckpt = "/tmp/siam_seg.ckpt"
            convert_checkpoint("smri_siam", f"{SIAM_MODEL_DIR}/fold_0/checkpoint_final.pth", ckpt)
        elif arm == "smri_mmunetvae":
            from asparagus_bridge.models_smri_mmunetvae import convert_mmunetvae_checkpoint
            ckpt = "/tmp/mmunetvae_seg.ckpt"
            convert_mmunetvae_checkpoint("/fomo26/weights/mmunetvae/fomo25_mmunetvae_pretrained.ckpt", ckpt)
        elif arm == "smri_fomo60k":
            from asparagus_bridge.models_smri_fomo60k import convert_fomo60k_checkpoint
            ckpt = "/tmp/fomo60k_seg.ckpt"
            convert_fomo60k_checkpoint(
                "/fomo26/weights/fomo60k_pkoutsouvelis/combined_regular-step=200000.ckpt", ckpt)
        elif arm == "scratch_nnunet":
            ckpt = None
        else:
            raise ValueError(f"unsupported seg arm {arm}")
    except Exception:
        import traceback
        return {"arm": arm, "fold": fold, "ok": False, "lesion_dice": None,
                "stderr_tail": f"ckpt conversion failed:\n{traceback.format_exc()}"[-2000:]}

    tag = arm[len("smri_"):] if arm.startswith("smri_") else arm
    rundir = f"/fomo26/models/seg_{tag}_isles22_fold{fold}{'_debug' if debug else ''}"
    cmd = ["python", "-m", "asparagus.pipeline.run.finetune_seg",
           f"task={TASK}", f"+model={arm}"]
    if ckpt:
        cmd += [f"checkpoint_path={ckpt}", "training.load_decoder=False"]
    cmd += ["data.train_split=split_80_10_10", "data.test_split=TEST_80_10_10", f"data.fold={fold}"]
    if debug:  # fast green check: exercises data+label load, train step, sliding-window test inference
        cmd += ["training.epochs=6", "training.patch_size=[64,64,64]", "training.batch_size=1",
                "training.train_batches_per_epoch_per_device=5", "training.val_batches_per_epoch_per_device=3",
                "training.warmup_epochs=1", "training.decoder_warmup_epochs=1", "training.check_val_every_n_epoch=1"]
    else:
        cmd += ["training.epochs=200", "training.patch_size=[128,128,128]", "training.batch_size=2",
                "training.train_batches_per_epoch_per_device=50", "training.val_batches_per_epoch_per_device=20",
                "training.warmup_epochs=10", "training.decoder_warmup_epochs=10", "training.check_val_every_n_epoch=5"]
    cmd += ["hardware.num_workers=8", "logger.wandb_logging=False", f"hydra.run.dir={rundir}"]
    proc = subprocess.run(cmd, env=env, cwd="/workspace/smri-fm", capture_output=True, text=True)
    ok = proc.returncode == 0
    pred = f"{rundir}/predictions/{TASK}__TEST_80_10_10__best.json"
    dice = None
    if os.path.exists(pred):
        dice = json.load(open(pred))["mean"]["1"]["dice"]
    vol.commit()
    return {"arm": arm, "fold": fold, "ok": ok, "rundir": rundir,
            "lesion_dice": dice, "stderr_tail": proc.stderr[-2000:] if not ok else ""}


@app.local_entrypoint()
def main(arms: str = "", folds: str = "", debug: int = 0):
    A = arms.split(",") if arms else ALL_ARMS
    F = [int(x) for x in folds.split(",")] if folds else ALL_FOLDS
    jobs = [(a, f, debug) for a in A for f in F]
    print(f">>> launching {len(jobs)} ISLES22 seg runs on Modal A100 (arms={A}, folds={F})")
    results = list(run_one.starmap(jobs))

    # aggregate FM vs scratch
    import statistics as st
    by_arm: dict[str, list[float]] = {}
    print(f"\n>>> {sum(r['ok'] for r in results)}/{len(results)} OK")
    for r in results:
        flag = "OK " if r["ok"] else "FAIL"
        d = f"{r['lesion_dice']:.4f}" if r["lesion_dice"] is not None else "—"
        print(f"  {flag} {r['arm']:16s} fold{r['fold']}  lesion Dice {d}")
        if r["ok"] and r["lesion_dice"] is not None:
            by_arm.setdefault(r["arm"], []).append(r["lesion_dice"])
        if not r["ok"]:
            print("       " + r["stderr_tail"].replace("\n", "\n       ")[-500:])
    print("\n=== per-arm lesion Dice (mean±std over folds) ===")
    means = {}
    for arm, v in by_arm.items():
        means[arm] = st.mean(v)
        sd = st.pstdev(v) if len(v) > 1 else 0.0
        print(f"  {arm:16s} {st.mean(v):.4f} ± {sd:.4f}  (n={len(v)})")
    sc = means.get("scratch_nnunet")
    if sc is not None:
        print("\n=== pretraining gain over scratch (Δ mean Dice) ===")
        for arm, m in means.items():
            if arm != "scratch_nnunet":
                print(f"  {arm:16s} {m - sc:+.4f}")
    json.dump(results, open("/tmp/isles22_seg_modal_results.json", "w"), indent=2)
    print("\n>>> wrote /tmp/isles22_seg_modal_results.json")
