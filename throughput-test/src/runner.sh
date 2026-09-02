#!/usr/bin/env bash
# runner.sh — Mac-side entrypoint: instance up -> stage -> setup -> sweep ->
# collect -> fetch -> html -> instance down. Every knob is an env var so any
# brev GPU type works (DS_INSTANCE=rtx-pro-6000 DS_HOST_ALIAS=... runner.sh).
#
#   DS_TAG=t4-baseline          run tag (default t4-baseline)
#   DS_N_LIST=1,2,4,...         sweep points
#   DS_WARMUP / DS_MEASURE      per-N windows (default 20/30 s)
#   KEEP_UP=1                   leave the instance running afterwards
#   DRY_RUN=1                   preflight only: checks + plan, no instance work
#   MAX_RUNTIME_SEC=...         sweep cap (guard rail)
set -uo pipefail
cd "$(dirname "$0")/.."

DS_INSTANCE="${DS_INSTANCE:-deepstream-t4}"
DS_TAG="${DS_TAG:-t4-baseline}"
DS_N_LIST="${DS_N_LIST:-1,2,4,8,16,24,32,48,64,96,128,160}"
DS_WARMUP="${DS_WARMUP:-20}"
DS_MEASURE="${DS_MEASURE:-30}"
KEEP_UP="${KEEP_UP:-0}"
DRY_RUN="${DRY_RUN:-0}"
MAX_RUNTIME_SEC="${MAX_RUNTIME_SEC:-}"

log() { printf '[runner] instance=%s tag=%s %s\n' "$DS_INSTANCE" "$DS_TAG" "$*" >&2; }
die() { printf '[runner] ERROR: %s\n' "$*" >&2; exit 1; }

. "$(dirname "$0")/remote.sh"

# DS_* knobs above must be exported before remote.sh reads them
export DS_INSTANCE DS_TAG DS_N_LIST DS_WARMUP DS_MEASURE

STAGE_METHOD="?"
T0=$(date +%s)

elapsed_s() { echo $(( $(date +%s) - T0 )); }

preflight() {
  log "preflight: selftest + brev CLI"
  python3 src/selftest.py || die "selftest failed"
  command -v brev > /dev/null || die "brev CLI not on PATH"
  command -v python3 > /dev/null || die "python3 not on PATH"
}

wait_ssh() {
  local i
  for i in $(seq 1 20); do
    if host_ssh "echo ssh-ok" > /dev/null 2>&1; then
      log "ssh up after $((i * 15))s"
      return 0
    fi
    sleep 15
  done
  die "ssh never came up (~5 min)"
}

run_setup() {
  log "running setup_env.sh in-container (idempotent)"
  host_ssh "sudo docker exec -i '$DS_CONTAINER' bash -s" < src/setup_env.sh \
    || die "setup_env.sh failed"
}

run_sweep() {
  log "starting sweep (python per $DS_WORK/throughput/env/python.txt)"
  local py
  py=$(host_ssh "sudo docker exec '$DS_CONTAINER' bash -c 'cat $DS_WORK/throughput/env/python.txt'") \
    || die "cannot read recorded python path"
  host_ssh "sudo docker exec '$DS_CONTAINER' bash -c '\
      $py $DS_WORK/throughput-test/src/sweep.py \
      --tag $DS_TAG --n-list $DS_N_LIST \
      --warmup $DS_WARMUP --measure $DS_MEASURE \
      --work $DS_WORK/throughput \
      ${MAX_RUNTIME_SEC:+--max-runtime-sec $MAX_RUNTIME_SEC} \
      2>&1 | tee -a /tmp/throughput-sweep-$DS_TAG.log'" \
    || die "sweep failed"
}

run_collect() {
  log "collecting summary in-container"
  host_ssh "sudo docker exec '$DS_CONTAINER' bash -c '\
      python3 $DS_WORK/throughput-test/src/collect.py \
      --run-dir $DS_WORK/throughput-test/runs/$DS_TAG'" \
    || die "collect failed"
}

fetch_and_build() {
  if [ "$STAGE_METHOD" = "bind" ]; then
    fetch_runs "$DS_TAG"
  else
    fetch_runs_docker_cp "$DS_TAG"
  fi
  log "building deliverable html"
  python3 src/build_html.py --run-dir "runs/$DS_TAG" || die "build_html failed"
}

preflight
if [ "$DRY_RUN" = "1" ]; then
  log "DRY_RUN: plan only"
  log "  tag=$DS_TAG n_list=$DS_N_LIST warmup=$DS_WARMUP measure=$DS_MEASURE"
  log "  instance=$DS_INSTANCE alias=$DS_HOST_ALIAS"
  log "  keep_up=$KEEP_UP max_runtime_sec=${MAX_RUNTIME_SEC:-none}"
  exit 0
fi

ensure_running
wait_ssh
find_container

if stage_probe; then
  stage_code
else
  log "bind probe failed; using docker-cp fallback"
  stage_code_docker_cp
fi
log "stage method: $STAGE_METHOD"

run_setup
run_sweep
run_collect

# archive the tracker yml actually used (provenance: it lives outside runs/)
host_ssh "sudo docker exec '$DS_CONTAINER' cp /work/throughput/cfg/tracker.yml \
    /work/throughput-test/runs/$DS_TAG/tracker.yml.used" \
  || log "WARN: tracker.yml.used archive skipped"
fetch_and_build

if [ "$KEEP_UP" = "1" ]; then
  log "KEEP_UP=1 -> leaving $DS_INSTANCE RUNNING (remember: ~\$/hr)"
else
  stop_instance
fi
log "done in $(elapsed_s)s — runs/$DS_TAG + throughput-test-v1.html"