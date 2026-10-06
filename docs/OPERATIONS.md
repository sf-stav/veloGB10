# Operating a long-running veloGB10 deployment

This is the operational recipe for multi-hour / 24 h-class deployments: how to keep the head
alive, what the gauges mean, which alerts are real, and how to read an exit. Everything here is
engine flags + standard Linux — the engine reads no environment variables, and none of these
examples use any.

## 1. Why the TP head needs a supervisor

Every TP head failure path exits the process: a scheduler failure exits `70`, the host-memory
watchdog exits `3`. The **node** re-arms by itself (its resident supervisor re-arms per head
session — start it once and forget it). The **head** has no in-process supervisor: one abort or
panic after hours means head down, node re-armed and waiting — to a user, "inference stopped".
A process supervisor on the head closes that hole; `--exit-on-fatal` (default since v0.7.3;
below) makes the exit prompt and supervisor-friendly instead of lingering in a dead state.

## 2. systemd units (examples — adjust paths, ports and flags)

The binary needs its `src/ptx/` directory next to it (kernel artifacts load relative to the
working directory), so point `WorkingDirectory` at the unpacked release directory.

Head unit (`/etc/systemd/system/velogb10-head.service`):

    [Unit]
    Description=veloGB10 TP head
    After=network-online.target

    [Service]
    User=<user>
    WorkingDirectory=/opt/velogb10
    ExecStart=/opt/velogb10/gb10_inference --server --model-dir /opt/models/<model> --host 0.0.0.0 --port 9000 --tp 2 --nodes <node-ip>:29500 --exit-on-fatal
    Restart=on-failure
    RestartSec=5

    [Install]
    WantedBy=multi-user.target

Node unit (`/etc/systemd/system/velogb10-node.service`), on each peer:

    [Unit]
    Description=veloGB10 TP node supervisor
    After=network-online.target

    [Service]
    User=<user>
    WorkingDirectory=/opt/velogb10
    ExecStart=/opt/velogb10/gb10_inference --node --host 0.0.0.0 --port 29500
    Restart=always
    RestartSec=5

    [Install]
    WantedBy=multi-user.target

`--exit-on-fatal` (EXL3 server, **default ON since v0.7.3**; `--exit-on-fatal off` opts out):
on a sticky CUDA error or scheduler-thread panic, finish the in-flight request with an error,
then exit `70` for the supervisor to restart — instead of serving every later request a 503
from a DEAD engine. The node never needs it (it re-arms). The unit examples above pass the flag
explicitly for readability; with the v0.7.3 default it can be omitted.

The release tarball's `run_tp_server.sh` / `run_tp_node.sh` wrap the same flags (`--help` on
both; `--dry-run` prints the exact engine command line) if you prefer scripts over units.

## 3. Log handling

Per-request log lines go through a bounded, non-blocking queue (4096 lines): a stalled log
*reader* no longer freezes the scheduler — lines are dropped and counted
(`velogb10_log_lines_dropped_total`), with one recovery summary line. Drops still lose
diagnostics, so:

- Prefer appending stdout/stderr to a file and rotating it daily (logrotate with copytruncate),
  or run under journald. Do **not** pipe stdout into a consumer that can stall — that is the
  pre-v0.7.3 failure mode the queue exists to survive.
- Expect roughly 1.5–2.5 MB/h (40–60 MB/day) at modest request rates.
- FATAL / boot / panic lines bypass the queue by design — they always reach stderr.
- Alert on `velogb10_log_lines_dropped_total > 0`: it means the collector stalled.

## 4. The gauges (head only — the node is log-only)

| gauge | meaning | alert idea |
|---|---|---|
| `velogb10_process_rss_bytes` | server RSS | +10%/h sustained over 3 h = leak investigation |
| `velogb10_thread_count` | OS threads | monotonic growth over 6 h = leak |
| `velogb10_open_fds` | open file descriptors | monotonic growth over 6 h = fd leak |
| `velogb10_mem_available_bytes` | box `MemAvailable` (the watchdog metric) | trending toward the `--mem-watchdog-gb` floor (default 5 GB) |
| `velogb10_mem_min_available_bytes` | minimum `MemAvailable` since boot | post-incident triage: how close did we get |
| `velogb10_scheduler_steps_total` | scheduler rounds completed | flat while requests are in flight = stall (grab logs + `/metrics`) |
| `velogb10_scheduler_busy` | 1 = scheduler work in progress; 0 = idle-waiting | alert **only together with the age gauge** |
| `velogb10_scheduler_last_step_age_seconds` | seconds since the last COMPLETED round or prefill chunk | alert on `scheduler_busy == 1 AND age > 120` **only**. With `busy == 0` the server is idle-waiting and a high age is EXPECTED — not a wedge, not an admission stall |
| `velogb10_graph_cache_entries` | CUDA graphs held | first-hours growth expected (key-space saturation); never flattening = new key classes |
| `velogb10_log_lines_dropped_total` | log lines dropped (reader stalled) | > 0 = fix the log collector |
| `velogb10_requests_rejected_total` | requests refused at `--max-waiting` | sustained growth = clients outpacing lanes |
| `velogb10_streams_cancelled_backlog_total` | streams cancelled at `--stream-backlog-events` | > 0 = a client stopped reading mid-stream |
| `velogb10_prompt_tokens_cached_total` | prompt tokens served from the prefix cache | information: cache hit rate vs `velogb10_prompt_tokens_total` |

Also still current: `velogb10_requests_waiting` (queue depth), `velogb10_engine_alive`,
`velogb10_max_batch`, the TTFT/end-to-end/prefill/decode histograms, the token and
speculation counters, and `velogb10_build_info`.

## 5. Admission and stream bounds (defaults)

- **`--max-waiting N`** (default 256; `0` = unlimited = pre-v0.7.3 behaviour): at N or more
  requests waiting beyond the lanes, the server answers `503` with `Retry-After: 5` instead of
  queueing unboundedly. The counter is the in-flight request guard — a request counts for its
  whole handler lifetime (a stalled reader stays counted), not only while queued for a lane,
  and a burst can overshoot by the requests admitted between the check and the send.
- **`--stream-backlog-events N`** (default 65536 ≈ 11 min at 100 tok/s; `0` = unlimited): a
  stream whose UNCONSUMED event backlog exceeds N is cancelled exactly like a client
  disconnect (lane freed; under TP the Cancel ships to the mirrors). The count is exact — the
  channel send-minus-receive counter, not a guess.

## 6. Exit codes

| code | meaning |
|---|---|
| 70 | engine abort: TP scheduler failure; fatal CUDA error (default; `--exit-on-fatal off` keeps a DEAD engine instead); scheduler-thread panic |
| 3 | host-memory watchdog: `MemAvailable` below `--mem-watchdog-gb` for 400 ms — the box survives, the process dies |
| 10 | TP abort code (in logs): dead peer detected in an exchange; the process exit is 70/3 |
| 11 | device-side exchange deadline (in logs, "status 11"); process exit is 70 |
| 6 | proxy/watchdog abort code (in logs, 10 s); process exit is 70 |

Post-incident triage: the exit status (`systemctl show -p ExecMainStatus` or the shell) plus
`grep -nE "panicked|state DEAD|watchdog|abort|FATAL"` on **both** ranks' logs.

## 7. Soak watch-list

Every minute, record: a full `/metrics` scrape to a dated file; `ps -o pid,etime,nlwp,rss -C
gb10_inference` on every rank box; the fd count of the head pid; GPU utilization+memory;
`free -g`; the tails of head AND node logs. Failure thresholds: RSS +10%/h sustained;
fds/threads monotonic; `requests_waiting` > 4× `--max-batch` sustained;
`scheduler_busy == 1 AND scheduler_last_step_age_seconds > 120` with work in flight; any TP
abort; `velogb10_log_lines_dropped_total > 0`.

> Note: the systemd units above are reviewed examples, not soak-tested artifacts — mark them
> for verification on your first long run. The 11-min backlog figure assumes a 100 tok/s
> single stream; log-volume estimates come from a code audit, not a measured day.
