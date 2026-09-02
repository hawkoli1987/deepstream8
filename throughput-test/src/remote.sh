#!/usr/bin/env bash
# remote.sh — Mac-side library for driving the brev instance + DS container.
# Sourced by runner.sh. All knobs are env vars so any GPU instance type works:
#   DS_INSTANCE    brev instance name          (default deepstream-t4)
#   DS_HOST_ALIAS  ssh alias for the host VM   (default <DS_INSTANCE>-host)
#   DS_CONTAINER   container name              (default: auto-discovered)
#   DS_STAGE_HOST  host-side staging dir       (default /home/ubuntu/mvlab)
#   DS_WORK        container-side /work root   (default /work)
#   DS_PEM         ssh identity                (from ssh -G, no default needed)
set -euo pipefail

DS_INSTANCE="${DS_INSTANCE:-deepstream-t4}"
DS_HOST_ALIAS="${DS_HOST_ALIAS:-$DS_INSTANCE-host}"
DS_STAGE_HOST="${DS_STAGE_HOST:-/home/ubuntu/mvlab}"
DS_WORK="${DS_WORK:-/work}"
DS_TAG="${DS_TAG:-t4-baseline}"
WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_LOCAL="$WS_ROOT/runs"

log() { printf '[runner] %s\n' "$*" >&2; }
die() { printf '[runner] ERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- ssh plumbing
# Stale-ssh_config immunity: brev's generated ssh_config lags instance boot by
# 5+ min; override Hostname from the instance metadata (via `brev exec --host`)
# while keeping User/Port/IdentityFile from the stored alias config.
resolve_ip() {
  if [ -z "${RESOLVED_IP:-}" ]; then
    RESOLVED_IP=$(brev exec "$DS_INSTANCE" --host \
        "curl -s --max-time 5 http://169.254.169.254/latest/meta-data/public-ipv4" \
        2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | tail -1 || true)
    log "resolved host ip: ${RESOLVED_IP:-<none, falling back to alias>}"
  fi
}

host_ssh() {
  resolve_ip
  if [ -n "$RESOLVED_IP" ] && [ -f ~/.brev/ssh_config ]; then
    ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new \
        -o "Hostname=$RESOLVED_IP" -F ~/.brev/ssh_config "$DS_HOST_ALIAS" "$@"
  elif [ -n "$RESOLVED_IP" ]; then
    ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new \
        -o "Hostname=$RESOLVED_IP" "$DS_HOST_ALIAS" "$@"
  else
    ssh -o BatchMode=yes -o ConnectTimeout=15 "$DS_HOST_ALIAS" "$@"
  fi
}

scp_host() {
  resolve_ip
  local extra=()
  [ -n "$RESOLVED_IP" ] && extra+=(-o "Hostname=$RESOLVED_IP")
  [ -f ~/.brev/ssh_config ] && extra+=(-F ~/.brev/ssh_config)
  scp -o BatchMode=yes "${extra[@]}" "$@"
}

# In-container exec: stdin script (heredoc) or command string.
ctr() { host_ssh sudo docker exec -i "$DS_CONTAINER" bash -s; }
ctr_cmd() { host_ssh sudo docker exec "$DS_CONTAINER" bash -lc "$*"; }

find_container() {
  if [ -n "${DS_CONTAINER:-}" ]; then
    log "container (pinned): $DS_CONTAINER"
    return 0
  fi
  DS_CONTAINER=$(host_ssh \
    "sudo docker ps --format '{{.Names}} {{.Image}}' | grep -i deepstream | head -1 | cut -d' ' -f1" || true)
  [ -n "$DS_CONTAINER" ] || die "no running deepstream container found (docker ps)"
  log "container: $DS_CONTAINER"
}

# ------------------------------------------------------------ brev lifecycle
instance_state() {
  brev ls 2>/dev/null | awk -v n="$DS_INSTANCE" '$1==n{print $2; exit}'
}

ensure_running() {
  local st
  st=$(instance_state)
  case "$st" in
    RUNNING)
      log "$DS_INSTANCE already RUNNING";;
    STOPPING)
      log "$DS_INSTANCE STOPPING — poll only, never re-issue (stop takes 5-6 min)"
      while [ "$(instance_state)" = "STOPPING" ] ; do sleep 20; done
      st=$(instance_state)
      log "reached $st"
      if [ "$st" = "STOPPED" ] || [ -z "$st" ]; then
        log "starting $DS_INSTANCE..."
        brev start "$DS_INSTANCE" || die "brev start failed"
      fi
      ;;
    ""|STOPPED)
      log "starting $DS_INSTANCE..."
      brev start "$DS_INSTANCE" || die "brev start failed"
      ;;
    *)
      die "unexpected instance state: '$st'";;
  esac
  local waited=0
  while [ "$(instance_state)" != "RUNNING" ]; do
    sleep 15
    waited=$((waited + 15))
    [ "$waited" -ge 1200 ] && die "instance not RUNNING after 20 min"
  done
  log "$DS_INSTANCE RUNNING"
}

stop_instance() {
  local st
  st=$(instance_state)
  if [ "$st" != "RUNNING" ]; then
    log "skip stop: $DS_INSTANCE is $st"
    return 0
  fi
  log "stopping $DS_INSTANCE (single call, then poll 5-6 min)"
  brev stop "$DS_INSTANCE" || die "brev stop failed"
  local waited=0
  while [ "$(instance_state)" != "STOPPED" ]; do
    sleep 30
    waited=$((waited + 30))
    [ "$waited" -ge 720 ] && { log "still not STOPPED after 12 min; leaving it"; return 0; }
  done
  log "$DS_INSTANCE STOPPED"
}

# ------------------------------------------------------------------ staging
stage_probe() {
  host_ssh "touch '$DS_STAGE_HOST/.tprobe.$$' 2>/dev/null \
    && sudo docker exec '$DS_CONTAINER' test -f '$DS_WORK/.tprobe.$$'" \
    && { host_ssh "rm -f '$DS_STAGE_HOST/.tprobe.$$'"; return 0; }
  return 1
}

stage_code() {
  log "staging workstream -> $DS_STAGE_HOST/throughput-test"
  tar -C "$WS_ROOT" -czf - src README.md 2>/dev/null | \
    host_ssh "mkdir -p '$DS_STAGE_HOST/throughput-test' \
      && tar -xzf - -C '$DS_STAGE_HOST/throughput-test'"
  STAGE_METHOD="bind"
}

stage_code_docker_cp() {
  local tgz="/tmp/throughput-stage-$$.tgz"
  tar -C "$WS_ROOT" -czf "$tgz" src README.md
  scp_host "$tgz" "$DS_HOST_ALIAS:/tmp/$(basename "$tgz")" \
    || die "scp staging fallback failed"
  host_ssh "sudo docker cp /tmp/$(basename "$tgz") '$DS_CONTAINER':/tmp/ \
    && sudo docker exec '$DS_CONTAINER' mkdir -p '$DS_WORK/throughput-test' \
    && sudo docker exec '$DS_CONTAINER' tar -xzf /tmp/$(basename "$tgz") -C '$DS_WORK/throughput-test'"
  rm -f "$tgz"
  STAGE_METHOD="docker-cp"
}

# -------------------------------------------------------------------- fetch
fetch_runs() {
  local tag="$1"
  local remote_tgz="/tmp/throughput-runs-$tag.tgz"
  local host_dir="$DS_STAGE_HOST/throughput-test/runs"
  host_ssh "cd '$host_dir' && tar -czf '$remote_tgz' '$tag'" \
    || die "remote tar of runs/$tag failed"
  mkdir -p "$RUNS_LOCAL"
  scp_host "$DS_HOST_ALIAS:$remote_tgz" "$RUNS_LOCAL/" || die "scp fetch failed"
  tar -xzf "$RUNS_LOCAL/$(basename "$remote_tgz")" -C "$RUNS_LOCAL"
  log "fetched runs/$tag -> $RUNS_LOCAL"
}

# docker-cp variant (no /work bind): pull the tarball out of the container first
fetch_runs_docker_cp() {
  local tag="$1"
  local remote_tgz="/tmp/throughput-runs-$tag.tgz"
  host_ssh "sudo docker exec '$DS_CONTAINER' tar -czf /tmp/r.tgz -C '$DS_WORK/throughput-test/runs' '$tag' \
    && sudo docker cp '$DS_CONTAINER':/tmp/r.tgz '$remote_tgz'" \
    || die "docker-cp run extraction failed"
  mkdir -p "$RUNS_LOCAL"
  scp_host "$DS_HOST_ALIAS:$remote_tgz" "$RUNS_LOCAL/" || die "scp fetch failed"
  tar -xzf "$RUNS_LOCAL/$(basename "$remote_tgz")" -C "$RUNS_LOCAL"
  log "fetched runs/$tag -> $RUNS_LOCAL (docker-cp)"
}