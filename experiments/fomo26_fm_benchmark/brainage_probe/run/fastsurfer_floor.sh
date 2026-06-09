#!/usr/bin/env bash
# Morphometry floor: FastSurfer --seg_only over FOMO26 Task-3 T1w (494 scans).
# Subject IDs use the harness convention sub-XXX_ses-wave1 so the downstream
# feature extractor parses (subject=sub-XXX, session=ses-wave1).
#
#   bash fastsurfer_floor.sh           # all 494
#   LIMIT=2 bash fastsurfer_floor.sh   # smoke first 2
set -uo pipefail

IMAGE="${IMAGE:-fastsurfer:grace}"
RAW_ROOT="/data/datasets/fomo26/source/Task_3/Task_3/preprocessed"
FS_ROOT="${FS_ROOT:-/data/datasets/fomo26/morphometry/fastsurfer}"
LICENSE="${LICENSE:-$HOME/license.txt}"
FS_CHECKPOINTS="${FS_CHECKPOINTS:-$HOME/fs_checkpoints}"
LOG_DIR="${LOG_DIR:-$(cd "$(dirname "$0")/.." && pwd)/logs}"
LIMIT="${LIMIT:-0}"
mkdir -p "$FS_ROOT" "$LOG_DIR"

mapfile -t subs < <(ls "$RAW_ROOT" | grep '^sub-' | sort)
[ "$LIMIT" -gt 0 ] && subs=("${subs[@]:0:$LIMIT}")
echo "FastSurfer floor: ${#subs[@]} subjects -> $FS_ROOT (image $IMAGE)"

done=0; skip=0; fail=0
for sub in "${subs[@]}"; do
  sid="${sub}_ses-wave1"
  t1="$RAW_ROOT/$sub/ses-01/t1w.nii.gz"
  marker="$FS_ROOT/$sid/mri/aparc.DKTatlas+aseg.deep.mgz"
  [ -f "$marker" ] && { skip=$((skip+1)); continue; }
  [ -f "$t1" ] || { echo "MISS $t1"; fail=$((fail+1)); continue; }
  docker run --rm --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --cpus "${CPU_LIMIT:-6}" --memory "${MEM_LIMIT:-30g}" \
    --user "$(id -u):$(id -g)" \
    -v "$RAW_ROOT/$sub/ses-01":/data:ro \
    -v "$FS_ROOT":/output:rw \
    -v "$LICENSE":/opt/FastSurfer/license.txt:ro \
    -v "$FS_CHECKPOINTS":/opt/FastSurfer/checkpoints:ro \
    "$IMAGE" \
      --t1 /data/t1w.nii.gz --sid "$sid" --sd /output \
      --seg_only --parallel --fs_license /opt/FastSurfer/license.txt \
    > "$LOG_DIR/$sid.fs.log" 2>&1 \
    && done=$((done+1)) || { echo "FAIL $sid (see $LOG_DIR/$sid.fs.log)"; fail=$((fail+1)); }
  echo "[$((done+skip+fail))/${#subs[@]}] done=$done skip=$skip fail=$fail  last=$sid"
done
echo "FastSurfer floor complete: done=$done skip=$skip fail=$fail"
