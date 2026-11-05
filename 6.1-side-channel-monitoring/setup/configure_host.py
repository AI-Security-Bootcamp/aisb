"""Configure a dedicated, single-node Slurm capture host. Run with sudo.

Prerequisites: slurm-wlm, slurmdbd, munge, mariadb-server, nftables, PicoSDK.
This does not change the NVIDIA driver, SSH access, or the host's default routes.
"""
import argparse
import json
import os
from pathlib import Path
import pwd
import re
import secrets
import socket
import subprocess


def run(*args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def write(path, text, mode=0o644, owner=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(mode)
    if owner:
        account = pwd.getpwnam(owner)
        os.chown(path, account.pw_uid, account.pw_gid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, choices=["RCP120XS", "RCP60XS"])
    args = parser.parse_args()
    if os.geteuid() != 0:
        raise SystemExit("Run with sudo")
    host = socket.gethostname().split(".")[0]
    cluster = host.replace("-", "_")
    slurm = Path("/etc/slurm")
    config = slurm / "slurm.conf"
    if config.exists() and "AISB dedicated capture host" not in config.read_text():
        raise SystemExit("Existing Slurm configuration is not managed by this script")
    cpu = run("slurmd", "-C", capture_output=True).stdout.splitlines()[0]
    fields = dict(item.split("=", 1) for item in cpu.split() if "=" in item)
    topology = " ".join(f"{k}={fields[k]}" for k in (
        "CPUs", "Boards", "SocketsPerBoard", "CoresPerSocket", "ThreadsPerCore"
    ) if k in fields)
    memory = max(1024, int(fields["RealMemory"]) - 4096)
    for folder in ("/var/spool/slurmctld", "/var/spool/slurmd", "/var/log/slurm"):
        Path(folder).mkdir(parents=True, exist_ok=True)
        account = pwd.getpwnam("slurm")
        os.chown(folder, account.pw_uid, account.pw_gid)

    run("systemctl", "enable", "--now", "munge", "mariadb")
    dbd = slurm / "slurmdbd.conf"
    password = secrets.token_urlsafe(36)
    if dbd.exists():
        old = re.search(r"^StoragePass=(.+)$", dbd.read_text(), re.M)
        if old:
            password = old.group(1)
    # The database password is sent via stdin and stored only in the 0600 config.
    sql = ("CREATE DATABASE IF NOT EXISTS slurm_acct;\n"
           f"CREATE USER IF NOT EXISTS 'slurm'@'localhost' IDENTIFIED BY '{password}';\n"
           f"ALTER USER 'slurm'@'localhost' IDENTIFIED BY '{password}';\n"
           "GRANT ALL ON slurm_acct.* TO 'slurm'@'localhost';\nFLUSH PRIVILEGES;\n")
    run("mariadb", input=sql, stdout=subprocess.DEVNULL)
    write(dbd, f"""# AISB dedicated capture host
AuthType=auth/munge
DbdHost=localhost
DbdAddr=127.0.0.1
DbdPort=6819
SlurmUser=slurm
LogFile=/var/log/slurm/slurmdbd.log
PidFile=/run/slurmdbd.pid
StorageType=accounting_storage/mysql
StorageHost=localhost
StorageUser=slurm
StoragePass={password}
StorageLoc=slurm_acct
""", 0o600, "slurm")
    write(config, f"""# AISB dedicated capture host
ClusterName={cluster}
SlurmctldHost={host}(127.0.0.1)
SlurmUser=slurm
SlurmdUser=root
AuthType=auth/munge
StateSaveLocation=/var/spool/slurmctld
SlurmdSpoolDir=/var/spool/slurmd
SlurmctldPidFile=/run/slurmctld.pid
SlurmdPidFile=/run/slurmd.pid
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log
ProctrackType=proctrack/cgroup
TaskPlugin=task/cgroup,task/affinity
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory
SchedulerType=sched/backfill
GresTypes=gpu,scope
ReturnToService=2
SlurmctldTimeout=120
SlurmdTimeout=120
JobAcctGatherType=jobacct_gather/linux
JobAcctGatherFrequency=30
AccountingStorageType=accounting_storage/slurmdbd
AccountingStorageHost=127.0.0.1
AccountingStoragePort=6819
AccountingStorageTRES=gres/gpu,gres/scope
NodeName={host} NodeAddr=127.0.0.1 {topology} RealMemory={memory} Gres=gpu:1,scope:1 State=UNKNOWN
PartitionName=capture Nodes={host} Default=YES MaxTime=00:30:00 DefaultTime=00:05:00 State=UP OverSubscribe=EXCLUSIVE DefMemPerNode=64000
""")
    write("/etc/udev/rules.d/71-aisb-pico.rules", 'SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", ATTR{idVendor}=="0ce9", MODE="0660", GROUP="users", SYMLINK+="aisb-scope"\n')
    run("udevadm", "control", "--reload-rules")
    run("udevadm", "trigger", "--subsystem-match=usb", "--attr-match=idVendor=0ce9")
    run("udevadm", "settle")
    if not Path("/dev/aisb-scope").exists():
        raise SystemExit("No PicoScope USB device found for scope:1")
    write(slurm / "gres.conf", "Name=gpu File=/dev/nvidia0 Count=1\nName=scope File=/dev/aisb-scope Count=1\n")
    write(slurm / "cgroup.conf", "CgroupPlugin=autodetect\nConstrainCores=yes\nConstrainRAMSpace=yes\nConstrainDevices=yes\n")
    # Keep scheduler/database control ports private without changing SSH or Tailscale rules.
    write("/etc/aisb/slurm-firewall.nft", """table inet aisb_slurm {
 chain input {
  type filter hook input priority -10; policy accept;
  iifname != "lo" tcp dport { 6817, 6818, 6819, 3306 } reject
 }
}
""")
    write("/etc/systemd/system/aisb-slurm-firewall.service", """[Unit]
Description=Keep AISB Slurm control ports local
Before=slurmctld.service slurmd.service slurmdbd.service
[Service]
Type=oneshot
ExecStart=-/usr/sbin/nft delete table inet aisb_slurm
ExecStart=/usr/sbin/nft -f /etc/aisb/slurm-firewall.nft
RemainAfterExit=yes
[Install]
WantedBy=multi-user.target
""")
    # PicoSDK's vendor udev rule provides USB permissions. Declare host-specific scale.
    probe = {"probe": args.probe, "sensitivity_mV_per_A": 50 if args.probe == "RCP120XS" else 100,
             "quantity": "AC current; not total DC power", "conductor": "unconfirmed",
             "timing_calibration": None}
    if host == "amodo-gigabyte-3":
        probe["timing_calibration"] = {"offset_us": 109.812, "uncertainty_us": 100,
                                      "interval_ns": .4, "duration_s": .42, "serial": "12669/0029"}
    write("/etc/aisb/probe.json", json.dumps(probe, indent=2) + "\n")
    run("systemctl", "daemon-reload")
    run("systemctl", "enable", "--now", "aisb-slurm-firewall")
    run("systemctl", "restart", "slurmdbd")
    run("systemctl", "enable", "slurmdbd", "slurmctld", "slurmd")
    run("sacctmgr", "-i", "add", "cluster", cluster)
    run("sacctmgr", "-i", "add", "account", "workshop", "Description=AISB", "Organization=AISB")
    run("sacctmgr", "-i", "add", "user", "tenant", "Account=workshop")
    run("systemctl", "restart", "slurmctld", "slurmd")
    run("sinfo", "-o", "%N %P %t %G %l")
    print(json.dumps({"host": host, "cluster": cluster, "probe": probe}, indent=2))


if __name__ == "__main__":
    main()
