#!/usr/bin/env bash
# Waits for the ISLES22 seg-finetune prediction JSONs, reports lesion Dice + comparison.
set -uo pipefail
PY=/home/mhough/dev/smri-fm/.venv/bin/python
M=/data/datasets/fomo26/models
declare -A ARM=( [siam]=seg_siam_seg011_isles22_ischstroke [mmunetvae]=seg_mmunetvae_seg011_isles22_ischstroke )
pred() { find "$M/$1" -name "*TEST*best.json" 2>/dev/null | head -1; }

# wait up to ~8h for both prediction jsons
for i in $(seq 1 960); do
  s=$(pred "${ARM[siam]}"); m=$(pred "${ARM[mmunetvae]}")
  [ -n "$s" ] && [ -n "$m" ] && break
  sleep 30
done
echo "================ ISLES22 stroke-lesion seg (DWI+ADC, test n=25) ================"
printf "%-14s %-12s %-12s %-12s\n" "arm" "lesion Dice" "sensitivity" "precision"
for a in siam mmunetvae; do
  p=$(pred "${ARM[$a]}")
  if [ -n "$p" ]; then
    $PY -c "import json;d=json.load(open('$p'))['mean']['1'];print('%-14s %-12.4f %-12.3f %-12.3f'%('$a',d['dice'],d.get('sensitivity',0),d.get('precision',0)))"
  else echo "$a: still running / no prediction json"; fi
done
echo "-------------------------------------------------------------------------------"
echo "Reference: FOMO260K paper AMAES on ISLES22 = Dice 73.98 (3-modality DWI/ADC/FLAIR, full challenge protocol)"
echo "Team FOMO26 seg leaderboard for context: SEG009 meningioma 0.0, SEG010 trigeminal 0.18-0.28"
