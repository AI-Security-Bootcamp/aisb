# Public SSH through Tailscale Funnel

Status on October 8, 2026: **public SSH and viewer forwarding verified on both
hosts**. Both GPU hosts persistently forward TLS port 443 to their local SSH
server. Tests connected to public Funnel relay addresses, with TLS certificate
and SSH host-key verification; both viewers returned HTTP 200 through `ssh -L`.
[Cloudflare remains the backup](CLOUDFLARE.md).

On October 9, machine 3 additionally exposes the
[public waveform viewer](https://amodo-gigabyte-3.pony-regulus.ts.net:8443/) on
HTTPS port 8443. Port 443 retains the SSH TLS mapping. The viewer still runs
internally on port 6008; Funnel cannot publish that public port.

The added mapping is:

```bash
sudo tailscale funnel --bg --https=8443 http://127.0.0.1:6008
```

To remove only public viewer access later:

```bash
sudo tailscale funnel --https=8443 off
```

This preserves SSH on port 443 and the existing private viewer address.

The authoritative public DNS records appeared around 23:09 UTC. Some recursive
resolvers still cached the preceding empty answer, whose negative-cache TTL was
300 seconds. Initial configuration was around 22:50 UTC; machine 3's mapping
was refreshed once around 23:00 UTC. The tests used independently resolved
public addresses to avoid both that cache and private MagicDNS resolution.
Do not pin the relay IPs in participant configuration: those addresses can change.

## Enable on the server

The installed Tailscale 1.102.4 supports TLS-terminated TCP forwarding. In the
[tailnet policy editor](https://login.tailscale.com/admin/acls/file), merge this
object into the existing `nodeAttrs` array, preserving other entries:

```json
{
  "target": ["tag:server"],
  "attr": ["funnel"]
}
```

If no `nodeAttrs` array exists, create it with this object as its first entry.
This is the deployed policy: both GPU hosts have `tag:server`. It grants Funnel
capability to every device with that tag. It does not replace the rest of the
policy or change SSH authentication. To restrict future enablement to these
two hosts, Tailscale also supports
[IP addresses as node-attribute targets](https://tailscale.com/docs/reference/targets-and-selectors#ip-address).

After a tailnet administrator saves the policy, run on each GPU host:

```bash
sudo tailscale funnel --bg --tls-terminated-tcp=443 tcp://127.0.0.1:22
tailscale funnel status --json
```

If the CLI prints an enablement URL, complete that administrator action first.
Do not substitute raw `--tcp`: the public Funnel connection must speak TLS,
whereas the existing SSH server speaks SSH. Do not enable Proxy Protocol for
the unmodified SSH server. No reboot or SSH daemon change is needed.

This port-443 mapping exposes SSH. Existing SSH keys authenticate users; local forwarding
provides access to the viewer without a public HTTP endpoint. The `tenant`
account is an instructor administrator; participants should use their assigned
accounts and keys.

## Participant configuration

The following example requires an OpenSSL client supporting the verification
flags shown and a working CA trust store. Participants do not need a Tailscale
or cloudflared client for this route. Add to `~/.ssh/config`:

Accounts `participant-1` through `participant-20` exist on both hosts. Replace
`YOUR_USER` below with your assigned account. The
[complete participant configuration](../README.md#configure-ssh-once) includes
`IdentityFile` and matching Cloudflare backup aliases.

```sshconfig
Host aisb-gpu3-funnel
    HostName amodo-gigabyte-3.pony-regulus.ts.net

Host aisb-gpu5-funnel
    HostName amodo-gigabyte-5.pony-regulus.ts.net

Host aisb-gpu3-funnel aisb-gpu5-funnel
    Port 443
    ProxyCommand openssl s_client -quiet -connect %h:%p -servername %h -verify_hostname %h -verify_return_error
    ServerAliveInterval 30
    ServerAliveCountMax 3
    ForwardAgent no
```

```bash
ssh YOUR_USER@aisb-gpu3-funnel
ssh -N -L 6008:127.0.0.1:6008 YOUR_USER@aisb-gpu3-funnel
# Use a different local port to view machine 5 at the same time:
ssh -N -L 6009:127.0.0.1:6008 YOUR_USER@aisb-gpu5-funnel
```

Open `http://localhost:6008` while the forwarding connection is running. Verify
the SSH host key against the instructor's fingerprint; TLS certificate
verification does not replace SSH host-key verification.

## Verify the public path and latency

The October 8 public test used 40 echoes per host over persistent SSH connections.
Funnel median RTT was 429 ms for machine 3 and 428 ms for machine 5; the 95th
percentiles were 563 and 529 ms. This was measured from the workspace, not the
US participant venue. Private Tailscale and Cloudflare had similar median RTTs
in the concurrent comparison; see the [route comparison](README.md#participant-access-from-the-us).

1. Confirm the Funnel configuration forwards TLS port 443 to local SSH port 22.
2. From a machine **outside the tailnet**, verify public DNS, the TLS certificate,
   SSH authentication, and an HTTP 200 from the viewer through `ssh -L`.
   Inside a tailnet, MagicDNS may resolve the name to `100.117.43.5`; that would
   test a private connection instead of the public Funnel relay. An operator
   testing from inside the tailnet must use an independently resolved public
   Funnel edge address while retaining the original TLS SNI, certificate
   hostname verification, and SSH host-key verification.
3. Run the persistent-connection RTT benchmark from the participant venue:

   ```bash
   python3 benchmark_ssh.py --config ~/.ssh/config --output funnel-latency.json \
       --source 'US participant venue' --samples 40 aisb-gpu3-funnel
   ```

4. Compare median and 95th-percentile echo latency with the same host over the
   other routes. Avoid bulk transfers during this measurement. Connection
   startup time is recorded separately and is not the per-keystroke delay.

Machine 3-to-5 currently uses a slow DERP hop. Use each host's public endpoint
to avoid that extra hop. Recheck latency from the participant venue.
Large completed recordings and their viewer can be hosted in the US separately.

## Remove this mapping

Run on each host being retired:

```bash
sudo tailscale funnel --tls-terminated-tcp=443 off
tailscale funnel status --json
```

Remove the client alias if no longer needed. This leaves Tailscale connectivity,
SSH, Slurm, recordings, and the Cloudflare backup intact. Avoid `funnel reset`
if other Funnel mappings have since been added.

References: [Funnel requirements](https://tailscale.com/docs/features/tailscale-funnel),
[TLS TCP forwarding](https://tailscale.com/docs/reference/tailscale-cli/funnel),
[OpenSSL verification options](https://docs.openssl.org/3.6/man1/openssl-s_client/).
