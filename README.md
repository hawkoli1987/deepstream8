# deepstream8

DeepStream 8 workstreams. Each subdirectory is a self-contained piece of work.

| dir | what |
|-----|------|
| [`motion-compare/`](motion-compare/) | frame-by-frame comparison of four motion-detection algorithms (CPU/GPU greyscale absdiff, NvOFA, MOG2) on a fixed-camera clip, with a single-page interactive viewer |

## Not in this repo

The NVIDIA DeepStream 8 SDK source tree (`sources/`, `service-maker/`) may be
present locally alongside these workstreams as a build reference. It is
**gitignored** — it is NVIDIA-proprietary and has no public open-source release.
Re-pull it from an NGC container or the SDK installer when needed.

## Remote / SSH

`origin` → `git@github.com:hawkoli1987/deepstream8.git`. This repo pins
`core.sshCommand` to `~/.ssh/ed25519_r7i`; load that key into your agent
(`ssh-add ~/.ssh/ed25519_r7i`) before pull/push.
