# Running the scale-up (run_02) on RunPod

Three scripts here:
- `make_upload_tarball.sh` — build a clean repo tarball to upload (run on your Mac).
- `setup.sh` — one-time env setup on the pod (venv, CUDA torch, test suite).
- `run_all.sh` — train → generate → seq_eval → seq_audit for the top-3 families,
  then package results.

The scale-up config (`--scaleup`: d_model=256, layers=6, dim_ff=1024, max_len=512,
T=1000 ≈ 8M params, + EMA + held-out val early-stop) is already baked into
`run_all.sh`. You should not need to edit anything.

---

## 0. Provision a pod
RTX 4090 (or better), a PyTorch/CUDA 12.x template, ~20 GB disk. Note its SSH/web
terminal. Install `runpodctl` locally if you haven't (for file transfer).

## 1. On your Mac — build the clean upload tarball
```bash
cd ~/CS271/d3pm_implementation
bash runpod/make_upload_tarball.sh
```
Writes `../d3pm_upload.tar.gz` (~19 MB). It **excludes** `.venv/`, `comparison/`,
old `runs/`, PDFs, and all macOS junk (`.DS_Store`, `._*` AppleDouble) — so none of
that "apple garbage" reaches the pod. The script also prints a sanity check
confirming nothing junk is inside.

## 2. Upload to the pod
```bash
runpodctl send ../d3pm_upload.tar.gz
# -> prints a one-time code; on the POD run:
runpodctl receive <code-it-prints>
```
(Or use the pod's web file upload / `scp` if you prefer.)

## 3. On the pod — extract & set up (one time)
```bash
mkdir -p d3pm_implementation && tar -xzf d3pm_upload.tar.gz   # extracts d3pm_implementation/
cd d3pm_implementation
bash runpod/setup.sh            # venv + CUDA torch + deps; runs pytest (expect 68 pass)
```
If `setup.sh` ends with "CUDA OK" and the tests pass, you're ready.

## 4. Run the full pipeline (in tmux so it survives disconnects)
```bash
tmux new -s d3pm
bash runpod/run_all.sh 2>&1 | tee run_all.log
# detach with Ctrl-b then d ; reattach later with:  tmux attach -t d3pm
```
This trains/gens/evals all three families and writes `d3pm_results.tar.gz`
(checkpoints + synthetic + eval_results + log), already cleaned of junk on the way
out. Typical wall time on a 4090: a few hours (early-stop usually ends well before
the 150-epoch cap).

## 5. Download results
```bash
# on the POD:
runpodctl send d3pm_results.tar.gz      # prints a code
# on your MAC:
cd ~/CS271/d3pm_implementation
runpodctl receive <code>
```

## 6. On your Mac — unpack into run_02 and re-evaluate
```bash
mkdir -p runs/run_02_scaleup
tar -xzf d3pm_results.tar.gz -C runs/run_02_scaleup
# (optional) keep the tarball:  mv d3pm_results.tar.gz runs/run_02_scaleup/

# --- sequence-level table on the bigger model ---
.venv/bin/python -m d3pm.seq_eval  --family winwebsec --synthetic runs/run_02_scaleup/synthetic --out-dir runs/run_02_scaleup/eval_results
.venv/bin/python -m d3pm.seq_audit --family winwebsec --synthetic runs/run_02_scaleup/synthetic --out-dir runs/run_02_scaleup/eval_results
#   ...repeat for zbot, zeroaccess (or the run_all output already has them)

# --- downstream classifier augmentation (before/after vs run_01) ---
cd comparison
PYTHONPATH=lib .venv/bin/python lib/seq_classifier.py \
    --d3pm-run ../runs/run_02_scaleup --out results/seq_classifier_run02.json

# --- embedding parity comparison ---
PYTHONPATH=lib .venv/bin/python lib/bridge_d3pm_sequences.py --d3pm-run ../runs/run_02_scaleup
PYTHONPATH=lib .venv/bin/python lib/evaluate.py --variant d3pm \
    --compare-against ddpm_baseline_report.json --out results
```
Then compare run_02 vs run_01 on the same tables. Per the stop-rule in
`../FINDINGS.md`: if seq-level 2/3-gram F1 and the augmentation lift don't improve
by ~10–15% relative, stop tuning the generator.

---

### Notes / gotchas
- **"Apple garbage":** macOS sprinkles `.DS_Store` and `._*` AppleDouble files;
  `tar` on a Mac also embeds `._` resource forks unless `COPYFILE_DISABLE=1`. Both
  upload and result tarballs set that and `--exclude` those patterns, so you won't
  ship or receive them. To purge any that exist:
  `find . -name .DS_Store -delete; find . -name '._*' -delete`.
- `setup.sh` pins torch 2.12 cu124 with a fallback to the latest cu124 wheel.
- Everything is reproducible from the upload tarball; the pod is disposable.
