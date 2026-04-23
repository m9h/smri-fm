#!/usr/bin/env bash
# Self-contained shell tests for the DLBS morphometry benchmark scripts.
# Run: bash test_shell.sh
# Exit 0 iff all tests pass.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$HERE/../scripts"
PASS=0
FAIL=0

red()   { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }

ok()   { green "  PASS: $*"; PASS=$((PASS+1)); }
fail() { red   "  FAIL: $*"; FAIL=$((FAIL+1)); }

# ---------------------------------------------------------------------
# Regression: smoke_all_subject.sh must reference images that actually
# exist locally. Bug 2026-04-22: after renaming grace→arm, overnight.sh
# silently failed every subject because docker pulls for fastsurfer-grace
# raised "pull access denied" and pipefail wasn't set.
# ---------------------------------------------------------------------
test_smoke_uses_arm_images() {
    local script="$SCRIPTS/smoke_all_subject.sh"
    [[ -f "$script" ]] || { fail "smoke script missing"; return; }
    if grep -q 'fastsurfer-grace\|t1prep-grace' "$script"; then
        fail "smoke_all_subject.sh still references -grace image names"
    else
        ok "smoke_all_subject.sh uses -arm image names"
    fi
}

test_smoke_referenced_images_exist() {
    local script="$SCRIPTS/smoke_all_subject.sh"
    local images
    images=$(grep -oE '(fastsurfer|t1prep|medarc)-(arm|grace):latest' "$script" | sort -u)
    local missing=0
    for img in $images; do
        if ! docker image inspect "$img" >/dev/null 2>&1; then
            fail "image '$img' referenced by smoke_all_subject.sh not available locally"
            missing=$((missing+1))
        fi
    done
    (( missing == 0 )) && ok "all images referenced by smoke_all_subject.sh exist"
}

# ---------------------------------------------------------------------
# Regression: overnight.sh's per-subject PASSED/FAILED detection must
# correctly surface failures. Without `set -o pipefail`, the pipe
# `bash smoke.sh | tee | > /dev/null` always returns 0.
# ---------------------------------------------------------------------
test_overnight_has_pipefail() {
    local overnight=/data/mhough/tmp/overnight.sh
    if [[ ! -f "$overnight" ]]; then
        ok "overnight.sh not present here (ok: it lives on NAS)"
        return
    fi
    if grep -qE 'set.*pipefail|set -[a-z]*o pipefail' "$overnight"; then
        ok "overnight.sh has pipefail enabled"
    else
        fail "overnight.sh lacks pipefail — failures will be silently passed"
    fi
}

# ---------------------------------------------------------------------
# Smoke: smoke_all_subject.sh bails with non-zero on a nonexistent subject.
# ---------------------------------------------------------------------
test_smoke_fails_on_bogus_subject() {
    local script="$SCRIPTS/smoke_all_subject.sh"
    [[ -f "$script" ]] || { fail "smoke script missing"; return; }
    local out
    out=$(bash "$script" ds004856 sub-DEFINITELYNOTREAL 2>&1)
    local rc=$?
    if (( rc == 0 )); then
        fail "smoke_all_subject.sh returned 0 for a nonexistent subject"
    else
        ok "smoke_all_subject.sh exits non-zero for nonexistent subject (rc=$rc)"
    fi
}

# ---------------------------------------------------------------------
# t1prep_longitudinal.sh uses a file path (not directory) for --long-data
# (regression for the upstream T1Prep CLI semantics bug found 2026-04-22).
# ---------------------------------------------------------------------
test_t1prep_long_uses_file_path() {
    local script="$SCRIPTS/t1prep_longitudinal.sh"
    [[ -f "$script" ]] || { fail "t1prep_longitudinal.sh missing"; return; }
    # Must reference _desc-realigned.nii.gz as the --long-data target,
    # not the directory /scratch on its own.
    if grep -qE '\-\-long-data.*(REALIGNED_NAME|_desc-realigned)' "$script"; then
        ok "t1prep_longitudinal.sh passes --long-data <realigned-file>"
    else
        fail "t1prep_longitudinal.sh might be passing a dir to --long-data"
    fi
}

# ---------------------------------------------------------------------
# Run all.
# ---------------------------------------------------------------------
echo "== shell tests =="
test_smoke_uses_arm_images
test_smoke_referenced_images_exist
test_overnight_has_pipefail
test_t1prep_long_uses_file_path
test_smoke_fails_on_bogus_subject
echo ""
echo "$PASS passed, $FAIL failed"
exit $(( FAIL > 0 ? 1 : 0 ))
