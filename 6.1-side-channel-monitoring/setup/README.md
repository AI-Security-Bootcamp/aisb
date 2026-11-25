# Instrument-host setup and validation

This operator setup supports the section's observable engineering outcomes:
submit an exclusive GPU/scope job, obtain unclipped ADC samples, retain the actual
probe scale, and associate activity with CUDA execution. It is provided
infrastructure, not an additional participant coding exercise.

The two independent Slurm clusters use partition `capture`, one `gpu` resource,
one `scope` resource, five-minute default jobs, and a 30-minute maximum. Scope
ownership is serialized by scheduling plus the SDK adapter's cooperative lock.
Munge authenticates scheduler traffic; local MariaDB/slurmdbd provide `sacct`.
The setup blocks scheduler/database control ports on non-loopback interfaces.

## Provision a prepared H200/Pico host

The vendor-provided NVIDIA driver, Pico `libpsospa.so`, and Python 3.14 are already
installed. Preserve the driver and kernel. Do not reboot these hosts as part of
provisioning; the first host has a reported SSD enumeration issue.

Install `slurm-wlm slurmdbd munge mariadb-server git rsync python3-venv nftables`.
Clone this repo into `/home/tenant/aisb`. Create
`/home/tenant/envs/power-inference`, using CUDA Torch 2.9.1+cu128,
Transformers 4.55.4, NumPy, SciPy, Matplotlib, termcolor, `nvidia-ml-py`, pytest,
black, and libcst. Cache `Qwen/Qwen3-0.6B` at
`/home/tenant/experiments/pico-access/model/Qwen3-0.6B` before class. Model loading
inside the exercise is local-only.

Configure the correct probe on each host:

```bash
sudo python3 setup/configure_host.py --probe RCP60XS   # amodo-gigabyte-5
# Use --probe RCP120XS on amodo-gigabyte-3.
```

The script derives CPU/memory topology from `slurmd -C`, declares a stable scope
device alias through udev, and creates `/etc/aisb/probe.json`. It refuses to replace
an existing Slurm configuration not marked as managed by this setup. Database
credentials remain in a mode-0600 `slurmdbd.conf`; SSH keys and credentials are not
part of the repo or viewer distribution.

`tenant` is the existing administrator account. Participants use the accounts
described below, without administrative sudo access.

## Participant accounts

Both hosts have `participant-1` through `participant-20`, with UIDs 2101–2120.
Each has a mode-0700 home, a writable source copy at `~/aisb`, and a Slurm
association with default account `workshop`. The copies contain the exercise
files, without the administrator's credentials or Git metadata. Existing
participant workspaces are preserved when provisioning runs again.

To provision a prepared host, supply the workshop **public** key:

```bash
sudo python3 setup/provision_participants.py --public-key /path/to/participants.pub
```

The script creates group `aisb-participants` and grants it traversal of
`/home/tenant`, allowing access to the existing read-only environment and model
cache. It does not grant home-directory listing or access to `tenant/.ssh`.
The shared scope lock is root-owned `root:users`, mode 0660; a tmpfiles rule
preserves it across boots and excludes it from age-based cleanup. Keep that
single lock path across all copies of the capture helper.

Participants activate the shared environment with:

```bash
cd ~/aisb/6.1-side-channel-monitoring
source /home/tenant/envs/power-inference/bin/activate
```

The October 9 provisioning installed the supplied `aisb-day0-participants` key
in all 20 accounts on each server. Its fingerprint is
`SHA256:PcsZwBO1Rz8AdkYzG6bJpjHV1MmJXk/y1J+ipFkD/e4`. Anyone holding its private
key can select any participant account; use separate keys if identities must be
restricted to their assigned users. No participant has sudo access.

Provisioning records are at `/var/lib/aisb/participants-20261009.json` on each
host; the source template is `/opt/aisb/participant-template-20261009`.
See the [participant SSH instructions](../README.md#connecting-to-the-gpu-servers)
for Funnel and Cloudflare connection examples.

Validation on October 9 authenticated all 40 logins through public Funnel and
completed one Slurm job per login (`COMPLETED`, exit `0:0`). Each job checked its
user identity, writable workspace, shared model access, GPU visibility, scope
device access, and the shared capture lock. A separate `participant-1` job on
each host performed CUDA matrix multiplication and saved 25 million unclipped
scope samples at 0.4 ns. This verifies participant access; it does not resolve
the second machine's weak physical signal documented below. Participant SSH
and viewer forwarding also passed through both Funnel and Cloudflare.

## Acceptance test

From the repo root, use the prepared environment to build the reference:

```bash
source /home/tenant/envs/power-inference/bin/activate
./build-instructions.sh --force --reference 6.1-side-channel-monitoring/section1_solution.py
cd 6.1-side-channel-monitoring/viewer
python -m unittest -v test_serve.py
cd ..
sbatch --partition=capture --nodes=1 --ntasks=1 --gres=gpu:1,scope:1 \
  --exclusive --time=00:05:00 --job-name=hardware-validation \
  --output=/home/tenant/side-channel-runs/validation-%j.log \
  --wrap='python -u setup/validate_hardware.py --output /home/tenant/side-channel-runs/NEW-RUN'
```

Create the log parent directory first and choose a fresh output path each time.
The test runs the actual generated reference: idle acquisition, two unit tests,
model warm-up, continuous training capture, and profiled training capture. It then
merges the trace and checks clipping, sample count, probe scaling, CUDA layer
attribution, trailing margin, direct RMS arithmetic, and an active/idle RMS ratio
above three. NVIDIA power/utilization is logged independently at a requested
20 ms polling interval; this does not imply a 20 ms sensor update rate.

The output includes raw scope recordings, CUDA/Perfetto traces, `validation.json`,
`analysis.npz`, `nvidia-driver.csv`, and `validation.png`. A successful script plus
Slurm `COMPLETED` / `0:0` is the acceptance criterion. A weak or absent current
response is a failure even if CUDA itself runs.

`--publish /home/tenant/side-channel-data` additionally creates a separate, fresh
viewer catalog from the profiled recording. It contains only raw samples, a
summary, and the merged trace. Run the viewer with:

```bash
python viewer/serve.py --host HOST_TAILSCALE_IP --port 6008 \
  --recordings /home/tenant/side-channel-data/recordings \
  --traces /home/tenant/side-channel-data/traces
```

For a persistent service on the second host, create a small NumPy environment at
`/home/tenant/envs/pico` and install the supplied service:

```bash
echo 'AISB_VIEWER_HOST=100.106.202.30' | sudo tee /etc/aisb/viewer.env
sudo install -m 644 setup/aisb-waveform-viewer.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now aisb-waveform-viewer
```

The service only reads the prepared catalog and binds to the Tailscale address.
The first host already has a viewer on port 6008; preserve it rather than
installing a second service on that port. After publishing a new catalog, restart
the corresponding viewer to load it.

Both probes measure nominal AC current, not total DC GPU power. The RCP60XS uses
100 mV/A. Exact conductor coverage is still unconfirmed. Its scope has no verified
timing calibration yet, so the merger marks it uncalibrated and the viewer says so.

### Investigate a missing signal

`setup/diagnose_channels.py --output NEW-DIRECTORY`, submitted with the same
exclusive Slurm resource request, records all four channels during four short GPU
load pulses. It writes the raw ADC array, a plot in millivolts, and NVIDIA driver
readings. Using volts for this check avoids assuming which channel has a probe.
If no channel responds, check probe/integrator power, the scope cable, and which
conductor the coil surrounds before repeating the acceptance test.

To obtain diagnostic plots from an already captured acceptance run, without
running another GPU job, use `--existing-run ORIGINAL-RUN --output NEW-ANALYSIS`.
The optional `--publish` still works on a weak-signal recording: the viewer title
explicitly says **probe check required**, the report says `failed`, and the
command exits with status 1. Publication does not mean the hardware passed.

For a possible multi-second synchronization error, submit
`setup/long_sweep.py --output NEW-DIRECTORY` with the same Slurm request. It captures
all four channels continuously for 60 seconds at the nearest native interval to
5 microseconds, with 10 seconds of idle baseline, unequal 2/4/6-second GPU bursts,
5-second gaps, and 28 seconds of trailing idle time. It records both NVIDIA power
APIs (including `NVML_FI_DEV_POWER_INSTANT`), utilization, host clock boundaries,
and when the scope first reports completion. The returned interval and sample
count are checked against the SDK timebase and requested duration.

Inspect `rms.png`, `envelope.png`, and `lag-search.png` alongside `report.json`.
The script searches lags of ±20 seconds; the largest noise correlation alone is
not evidence of an offset. Look for all three unequal bursts at a consistent lag,
and compare with an identical capture on the known-working first instrument.
Readiness timing bounds gross acquisition delay but does not calibrate sample zero.
At this lower sample rate, the 30 MHz probe signal can alias: the 10 ms RMS and
min/max bins describe the retained native samples, not all high-frequency energy
or extrema. Raw ADC samples are retained for further inspection.

### Verification on October 8, 2026

Both hosts passed the eight viewer tests and executed the generated training
reference, including the current-conversion and optimizer-update tests. Both
produced 1.05 billion unclipped samples at 0.4 ns and a trace with 28 GPU layer
spans. Hardware acceptance additionally requires a workload-dependent signal:

| Host | Probe | Idle / active median 0.1 ms RMS | Hardware acceptance |
| --- | --- | --- | --- |
| amodo-gigabyte-3 | RCP120XS, 50 mV/A | 0.323 / 4.659 A | Passed; Slurm job 2, 38 seconds |
| amodo-gigabyte-5 | RCP60XS, 100 mV/A | 0.138 / 0.142 A | Failed; Slurm job 1, 39 seconds |

The second host's additional four-channel test reached 411 W in NVIDIA telemetry,
but no channel showed a corresponding response. Physical probe power, cable, and
conductor placement need checking. The viewer on port 6008 exposes this diagnostic
capture as **probe check required**, with uncalibrated timing. Acceptance artifacts
are under `/home/tenant/side-channel-runs/acceptance-20261008`; the second host's
saved-data analysis and four-channel diagnostic are in `analysis-20261008` and
`channels-20261008` alongside it. All validation allocations requested five minutes
and finished in under a minute; neither queue had a remaining job.

The later 60-second sweep tested the synchronization-delay hypothesis. Machine 5
job 3 used 199.808 kS/s, reached 594 W, and returned all samples without overflow.
Channel A stayed at 14.95 / 14.98 mV idle/active RMS of the retained samples.
Channel B showed a small repeatable decrease during load, near the noise level;
this does not establish probe placement or calibrated current. A ±20-second lag
search found no delayed large channel-A response.

The first machine's control succeeded at 99.968 kS/s (job 4), with channel A
increasing from 17.68 to 152.96 mV and correlation 0.9924 at zero lag. Both scopes
first reported completion about 60.010 seconds after `RunBlock` returned. These
observations do not support a many-second synchronization delay. Both successful
jobs took 68 seconds. The first machine's initial attempt failed during USB data
download after a device disconnect; `slurmd` alone was restarted while idle to
refresh its device registration, and the smaller transfer succeeded without a
reboot. Raw results remain in `long-sweep-20261008-182816` on machine 5 and
`long-sweep-control-retry-20261008-1834` on machine 3 under `side-channel-runs`.

## Participant access from the US

The October 8 check found roughly 500 ms to the London DERP relay, changing NAT
mapping, and relayed connections between the instrument hosts. A direct ping from
the second host to 1.1.1.1 was 26–78 ms. Private-LAN communication between the
hosts failed. These are observations from that check, not a latency guarantee.

First ask the provider for a routable/public address or an outbound tunnel path
that avoids the current relay. Tailscale's [performance guide](https://tailscale.com/docs/reference/troubleshooting/poor-performance-tailnet)
explains why direct connections generally perform better. Changing DNS or making
the same slow backend public will not remove transatlantic round trips.

Tailscale is the primary access route, with the existing Cloudflare Tunnels at
`gpu3.aisb.dev` and `gpu5.aisb.dev` retained as backups. See
[Cloudflare backup instructions](CLOUDFLARE.md). The backup requires client-side
`cloudflared`; it does not need a paid website plan. The unsuccessful Spectrum
attempt's Pro subscription is scheduled to return to Free on November 8, 2026.

Latency is the priority for interactive sessions. A 40-echo sample from the
workspace on October 8 used persistent authenticated SSH connections, with
42-byte requests, three warmup echoes, and no PTY:

| Route | Median RTT | 95th percentile |
| --- | --- | --- |
| Workspace to machine 3 over public Funnel | 429 ms | 563 ms |
| Workspace to machine 5 over public Funnel | 428 ms | 529 ms |
| Workspace to machine 3 over private Tailscale | 422 ms | 522 ms |
| Workspace to machine 5 over private Tailscale | 424 ms | 494 ms |
| Workspace to machine 3 over Cloudflare | 430 ms | 482 ms |
| Workspace to machine 5 over Cloudflare | 435 ms | 525 ms |

The six routes were measured concurrently after public Funnel verification.
Funnel did not show a material median latency improvement in this sample.
These are workspace measurements, not measurements from the US venue. In an
earlier sample, reaching machine 5 through Cloudflare and machine 3 had a median
of 1,069 ms. The machine 3-to-5 link used London DERP and took 419–565 ms in three Tailscale pings;
the additional bastion hop therefore matters. Prefer separate connections to
each machine for latency. Ordinary Internet ICMP from machine 3 to 1.1.1.1 took
25–44 ms in a separate sample. Different protocols and destinations prevent
attributing all SSH delay to one component, but the slow relay path deserves
investigation rather than assuming geographic distance explains everything.

[Tailscale Funnel](https://tailscale.com/docs/features/tailscale-funnel) provides
public access without a participant VPN client. It needs an
OpenSSL TLS wrapper in SSH's ProxyCommand; SSH port forwarding then works
normally. **Funnel is configured on both GPU hosts with persistent TLS port 443
forwarding to local SSH.** The `tag:server` node-attribute rule now permits both
hosts. **Both public routes passed authenticated SSH and viewer forwarding
checks**, with certificate and SSH host-key verification. Public DNS records
appeared around 23:09 UTC; some recursive resolvers retained cached empty
answers briefly afterward. See [Funnel setup and connection commands](FUNNEL.md).

Use `benchmark_ssh.py --config PATH --output results.json HOST [HOST ...]` to
repeat the small-message measurement from the actual participant venue. SSH
host aliases and identities belong in the supplied config file. The script
records connection/process startup separately from steady-state echo RTT.

The loopback viewer forwarders remain available independently of the chosen
SSH transport. Forward `127.0.0.1:6008` on the selected host with `ssh -L`; keep
Slurm and database ports private.

Host the CPU-only viewer and completed data near participants in the US, with
HTTPS and event authentication. Transfer each completed recording once, then serve
zoom/RMS queries and trace downloads from that copy. Editing can also happen in
a US workspace, with code synchronized and jobs submitted asynchronously to the
UK hosts. The physical acquisition stays in Sheffield; moving the viewer and
editing session removes repeated UK round trips from those interactions.
Publish the small Perfetto archive first (about 6.4 MB for this run), then copy the
2.1 GB raw waveform in the background. The viewer already supports trace-only
catalog entries while the raw samples are unavailable.
