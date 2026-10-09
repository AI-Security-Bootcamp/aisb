# 6.1 — Power side channels

**Suggested time:** 2 hours, working in pairs. The first 15 minutes cover the
measurement hardware and an idle capture.

**Exercises:** [Participant instructions](section1_instructions.md).

| Area | Prerequisites coming in | Learning outcomes |
| --- | --- | --- |
| Engineering | Create a PyTorch tensor with a specified shape and device; write a forward/loss/backward/optimizer step | Submit a standalone script to Slurm and inspect its logs/results; place acquisition boundaries around completed CUDA work; export a CPU/CUDA timeline; retain the metadata needed to combine independent recordings |
| ML | Explain what a batch, forward pass, backward pass, and parameter update represent | Identify these phases in a trace; distinguish batch-size and sequence-length effects on throughput and kernel shapes |
| Security | - | Test whether an observer of supply-current variation can infer model depth or workload shape; separate evidence from alternative explanations |
| Theory | - | Convert ADC counts to nominal current; explain bandwidth, sampling, RMS, and timing uncertainty; explain why a current probe cannot reveal total DC power in this setup |

## Background

- https://www.ti.com/lit/ta/ssztdb4/ssztdb4.pdf
- https://www.tek.com/en/documents/primer/oscilloscope-basics
- [slurm basics - `sbatch`](https://slurm.schedmd.com/sbatch.html)
- PyTorch [Optimization Loop](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#optimization-loop)

## Connecting to the GPU servers

Run these commands on your own laptop or workstation. Tailscale Funnel requires
OpenSSH and OpenSSL; you do not need to install Tailscale or join the tailnet.

| Route                      | Machine 3 | Machine 5 |
|----------------------------| --- | --- |
| Tailscale Funnel           | `amodo-gigabyte-3.pony-regulus.ts.net` | `amodo-gigabyte-5.pony-regulus.ts.net` |
| Cloudflare Tunnel (backup) | `gpu3.aisb.dev` | `gpu5.aisb.dev` |

### SSH Configuration

Add this to `~/.ssh/config`. Remember to replace `participant-1` with your assigned account.

```sshconfig
Host aisb-gpu3 aisb-gpu5 aisb-gpu3-cf aisb-gpu5-cf
    User participant-1
    IdentityFile ~/.ssh/YOUR_PRIVATE_KEY
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 3
    ForwardAgent no

Host aisb-gpu3
    HostName amodo-gigabyte-3.pony-regulus.ts.net

Host aisb-gpu5
    HostName amodo-gigabyte-5.pony-regulus.ts.net

Host aisb-gpu3 aisb-gpu5
    Port 443
    ProxyCommand openssl s_client -quiet -connect %h:%p -servername %h -verify_hostname %h -verify_return_error

Host aisb-gpu3-cf
    HostName gpu3.aisb.dev

Host aisb-gpu5-cf
    HostName gpu5.aisb.dev

Host aisb-gpu3-cf aisb-gpu5-cf
    Port 22
    ProxyCommand cloudflared access ssh --hostname %h
```

### Connect through Tailscale Funnel

With that configuration saved, connect to your assigned machine:

```bash
ssh aisb-gpu3 # (or) ssh aisb-gpu5
```

<details>
<summary>Direct SSH connection commands</summary>

```bash
ssh -i ~/.ssh/YOUR_PRIVATE_KEY -p 443 \
  -o 'ProxyCommand=openssl s_client -quiet -connect %h:%p -servername %h -verify_hostname %h -verify_return_error' \
  participant-1@amodo-gigabyte-3.pony-regulus.ts.net
```

</details>

<details>
<summary>Connecting through the Cloudflare backup</summary>

Install [cloudflared for your operating system](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/downloads/)
and make sure `cloudflared --version` works in your terminal. On macOS with
Homebrew, use `brew install cloudflared`; on Linux, use the package or binary
for your distribution and CPU architecture from that download page.

Use these to connect to the servers:

```bash
ssh aisb-gpu3-cf
ssh aisb-gpu5-cf
```

For more details, see the
[Cloudflare guide](setup/CLOUDFLARE.md) and
[client-side SSH documentation](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/use-cases/ssh/ssh-cloudflared-authentication/).
</details>

## Pre-captured data

You can also access pre-captured data here
[https://amodo-gigabyte-3.pony-regulus.ts.net:8443/](https://amodo-gigabyte-3.pony-regulus.ts.net:8443/).

If you cannot connect to the machines, or are self-studying this module, you can use these links to download the
capture data.

Download a Perfetto archive(trace.json.gz) and select **Open trace file** at
[ui.perfetto.dev](https://ui.perfetto.dev/)

| Recording | Perfetto download | Acquisition metadata |
| --- | --- | --- |
| Baseline training cycle | [trace.json.gz, 6.51 MB](https://traces.aisb.dev/side-channels/2026-10-09/reference/trace.json.gz) | [summary.json](https://traces.aisb.dev/side-channels/2026-10-09/reference/summary.json) |
| Small MoE: Granite-3.1-1B-A400M | [trace.json.gz, 8.28 MB](https://traces.aisb.dev/side-channels/2026-10-09/granite-moe/trace.json.gz) | [summary.json](https://traces.aisb.dev/side-channels/2026-10-09/granite-moe/summary.json) |

For the full raw waveform, download the entire capture file:

- [Baseline raw bundle, 1.26 GB](https://traces.aisb.dev/side-channels/2026-10-09/reference/reference-raw.aisb.zip)
- [Granite MoE raw bundle, 1.27 GB](https://traces.aisb.dev/side-channels/2026-10-09/granite-moe/granite-moe-raw.aisb.zip)

These contain the 1.05 billion samples at 0.4 ns across the original 420 ms,
matching calibration metadata, and the Perfetto trace. To view, start the
[local viewer](viewer/README.md#import-a-raw-recording), choose **Import recording**,
select the `.aisb.zip`, and click **Load bundle**. The waveform view can then
show min/max envelopes, RMS, or individual samples. Allow approximately 5 GB
of free disk space per import. The shared public viewer already has these
recordings; local import keeps your downloaded data on your own machine.

## Errata
Slurm uses the `workshop` account by default and records your participant username
as the job owner. Use `squeue -u "$USER"` to see your jobs.

| Host | Probe | Nominal sensitivity |
| --- | --- | --- |
| `amodo-gigabyte-3.pony-regulus.ts.net` | RCP120XS | 50 mV/A |
| `amodo-gigabyte-5.pony-regulus.ts.net` | RCP60XS | 100 mV/A |

Each host has its own single-node Slurm `capture` partition. Request
`--gres=gpu:1,scope:1 --exclusive`; a submission uses the GPU and scope on the
machine where you submit it. The default limit is five minutes and the maximum is
30 minutes. `sacct` retains job state and exit status. Use SSH for editing and
submission, and Slurm for GPU work and instrument access.

The scope helper reads the installed probe scale from `/etc/aisb/probe.json` and
copies it into the capture metadata. Always convert using that metadata. The
second machine's clock alignment is initially uncalibrated; it does not inherit
the first instrument's calibration.

For provisioning and the real-hardware acceptance test, see [host setup](setup/README.md).
The [portable viewer](viewer/README.md) runs separately from acquisition.
