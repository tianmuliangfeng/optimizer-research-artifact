# MECH-03 remote runbook

## 1. Synchronize code

The following local files must appear under the same relative paths on both
remote hosts:

```text
${SNM_REPO}\scripts\31_mech03_crossfit_shadow\README.md
${SNM_REPO}\scripts\31_mech03_crossfit_shadow\prediction_contract.json
${SNM_REPO}\scripts\31_mech03_crossfit_shadow\run_mech03.py
${SNM_REPO}\scripts\31_mech03_crossfit_shadow\mech03_worker.py
${SNM_REPO}\scripts\31_mech03_crossfit_shadow\analyze_mech03.py
${SNM_REPO}\scripts\31_mech03_crossfit_shadow\test_mech03_contract.py
${SNM_REPO}/commands/31_mech03_crossfit_shadow/20260727_mech03_r1_endpoint.sh
${SNM_REPO}/commands/31_mech03_crossfit_shadow/20260727_mech03_llama_host_endpoint.sh
```

Remote root:

```text
${SNM_REPO}
```

On each host, verify the frozen contract before running:

```bash
sha256sum \
  ${SNM_REPO}/scripts/31_mech03_crossfit_shadow/prediction_contract.json
```

Required value:

```text
9b56e112797103cfc8c98948850a50ba59d672255f305f8dce2c0f5941a25712
```

## 2. R1 host

Run:

```bash
set -o pipefail
cd ${SNM_REPO}
bash ${SNM_REPO}/commands/31_mech03_crossfit_shadow/20260727_mech03_r1_endpoint.sh \
  2>&1 | tee /tmp/mech03_r1_endpoint_20260727.log
```

The script runs smoke first, requires `mech03_manifest.json: passed=true`, and
only then starts formal.

## 3. LLaMA host

This can run at the same time as the R1-host command. GPT bridge and LLaMA124
run sequentially on the selected GPU:

```bash
set -o pipefail
cd ${SNM_REPO}
bash ${SNM_REPO}/commands/31_mech03_crossfit_shadow/20260727_mech03_llama_host_endpoint.sh \
  2>&1 | tee /tmp/mech03_llama_host_endpoint_20260727.log
```

## 4. Handoff

Successful scripts print the exact package path and SHA-256. Packages are
created below:

```text
${SNM_RESULTS_ROOT}/31_mech03_crossfit_shadow/handoff_packages
```

Return these two files:

```text
mech03_r1_endpoint_<timestamp>.tgz
mech03_llama_host_endpoint_<timestamp>.tgz
```

Also retain the two `/tmp/mech03_*_20260727.log` files until local audit is
complete. Do not run MECH-04 after a terminal `PASS`; the frozen analyzer and
review still come first.
