# Backup SSH access through Cloudflare

| Public hostname | Instrument host | Probe |
| --- | --- | --- |
| `gpu3.aisb.dev` | `amodo-gigabyte-3` | RCP120XS, 50 mV/A |
| `gpu5.aisb.dev` | `amodo-gigabyte-5` | RCP60XS, 100 mV/A; physical signal check still required |

Cloudflare is retained as a backup to Tailscale access. This backup uses
Cloudflare Tunnel and requires `cloudflared` on the client. Native public TCP/22
is not enabled. Cloudflare Access is not enabled:
authentication uses existing SSH authorized keys. Both SSH servers disable
password and keyboard-interactive login.

On October 8, 2026, an attempted Spectrum deployment upgraded `aisb.dev` to Pro
($25/month), but the API rejected the required `virtual_network_id` field with
error 10012. Basic Pro SSH support does not include this Tunnel-origin
configuration; Cloudflare's account team must enable the full Spectrum add-on.
A return to Free is scheduled for November 8, 2026, at 00:00 UTC, so Pro will
not renew. No Spectrum application or `bastion.aisb.dev` DNS record was created.
See [Cloudflare's entitlement restriction](https://developers.cloudflare.com/spectrum/reference/error-codes/#10012--unknown-field-in-request-json).

## Connect

Install [cloudflared](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/downloads/)
(`brew install cloudflared` on macOS). Add this to `~/.ssh/config`:

```sshconfig
Host gpu3.aisb.dev gpu5.aisb.dev
    ProxyCommand cloudflared access ssh --hostname %h
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

Use your assigned account and authorized SSH key:

```bash
ssh YOUR_USER@gpu3.aisb.dev
ssh YOUR_USER@gpu5.aisb.dev
```

Accounts `participant-1` through `participant-20` are provisioned on both hosts,
each with its own home and Slurm identity. Replace `YOUR_USER` with your assigned
account. The existing instructor account is `tenant`, with administrative sudo
access. See the [complete participant SSH configuration](../README.md#configure-ssh-once)
for key selection and both connection routes. Ordinary `scp`,
`rsync` over SSH, and VS Code Remote SSH use the same SSH configuration.

Keep these connections running to access both waveform viewers:

```bash
ssh -N -L 6008:127.0.0.1:6008 YOUR_USER@gpu3.aisb.dev
# In a second terminal:
ssh -N -L 6009:127.0.0.1:6008 YOUR_USER@gpu5.aisb.dev
```

Open <http://localhost:6008> or <http://localhost:6009>. The existing Tailscale
viewer URLs still work. The public hostnames carry SSH only; opening
`https://gpu3.aisb.dev` is not a viewer URL.

### Reach machine 5 through machine 3

Only the first host needs to be publicly reachable. With the ProxyCommand above
configured for `gpu3.aisb.dev`, add this alias for the second host:

```sshconfig
Host aisb-gpu5-via-gpu3
    HostName 100.106.202.30
    HostKeyAlias gpu5.aisb.dev
    ProxyJump YOUR_USER@gpu3.aisb.dev
    ForwardAgent no
```

Then run:

```bash
ssh -N -L 6009:127.0.0.1:6008 YOUR_USER@aisb-gpu5-via-gpu3
```

Machine 3 connects to machine 5 over Tailscale; the participant does not need a
Tailscale client for this route. The participant's SSH key authenticates to both
hosts without being copied to the bastion. The first hop still needs
`cloudflared` until native public SSH is provisioned.

A later 40-echo latency sample measured about 495 ms to machine 3 and 1,069 ms
to machine 5 through that bastion, versus 479 ms to machine 5 over Tailscale
without the bastion. The private hop used London DERP and added substantial
delay. Prefer a separate connection to each host when interactive latency
matters; a bastion remains useful when only one public endpoint is available.

## Deployed services

Each host runs `aisb-cloudflared.service`, enabled on boot, using cloudflared
2026.10.0. Its remotely managed ingress configuration routes its hostname to
`ssh://127.0.0.1:22`, with a final `http_status:404` rule. A proxied CNAME points
each public hostname to its own tunnel's `UUID.cfargotunnel.com`. No private
subnet route or public Slurm, database, or waveform HTTP endpoint was added.

`/etc/cloudflared/aisb-tunnel-token` is root-owned mode 0600. Systemd supplies this
per-tunnel credential through `LoadCredential` to an unprivileged dynamic user.
The Cloudflare management API token is not stored on either host and is not
needed to keep the tunnels running.

The viewer continues listening on its existing Tailscale address.
`aisb-viewer-forward.socket` listens only on `127.0.0.1:6008`; its corresponding
service forwards requests to that same machine's Tailscale listener. The upstream
is set in `/etc/aisb/viewer-forward.env`. This preserves existing URLs while
supporting the standard SSH `-L` commands above.

## Operator checks

```bash
systemctl is-active aisb-cloudflared aisb-viewer-forward.socket
curl --fail http://127.0.0.1:20241/ready
curl --fail http://127.0.0.1:6008/ --output /dev/null
sudo journalctl -u aisb-cloudflared --since '10 minutes ago'
```

The forwarder service starts on the first request; its socket should already be
active. Connector readiness alone does not test SSH or the viewer: also verify
authenticated public SSH and an HTTP request through `ssh -L` from a participant
machine. Verify host keys against the existing Tailscale connection or an
instructor-provided fingerprint.

Deployment checks on October 8 authenticated over both public hostnames with
pinned origin host keys and loaded each viewer through local SSH forwarding
(HTTP 200). Both connectors reported four healthy edge connections. A ten-echo
sample measured median SSH round trips of 498 ms (machine 3) and 534 ms (machine
5) through Cloudflare. Tailscale comparison samples were 536 ms and 178 ms,
respectively, with substantial variation. These short samples do not demonstrate
a consistent Cloudflare latency improvement.

The workspace resolver temporarily cached NXDOMAIN for the newly created machine
5 hostname. Cloudflare's public resolver and machine 3 resolved it normally; its
workspace smoke test used a temporary local DNS override to a verified Cloudflare
edge address, preserving TLS and SSH host-key verification. The override was
removed immediately after the test.

The tunnel changes the network path, not the physical location or 5G uplink.
Measure sustained SSH round trips and waveform downloads from the actual venue
before treating it as a latency improvement. Keep Tailscale available.

To reproduce the host configuration, supply a separate tunnel token and viewer
environment file on each host, then install the supplied units into
`/etc/systemd/system/`. Run `systemctl daemon-reload` and
`systemctl enable --now aisb-cloudflared aisb-viewer-forward.socket`.
Do not copy one host's tunnel token to the other: that creates replicas of the
same tunnel and can send sessions to the wrong instrument.

Cloudflare's [client-side SSH guide](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/use-cases/ssh/ssh-cloudflared-authentication/)
also describes adding Access identity policies later. To remove the participant
binary, use [Spectrum with a virtual network origin](https://developers.cloudflare.com/spectrum/get-started/#create-a-spectrum-application-using-a-virtual-network-origin)
after Cloudflare confirms the full Spectrum entitlement, or obtain a public
TCP endpoint for machine 3 from the provider. Increasing the website plan alone
does not resolve the observed entitlement error. A single native SSH bastion
would suffice for both machines.
