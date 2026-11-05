"""Provision the 20 workshop logins on a prepared AISB instrument host.

Run as root with a participant public-key file. Existing home directories and
authorized keys are preserved. A validation public key may be added temporarily
and must be removed after the login checks. No private key is needed here.
"""

import argparse
import grp
import json
import os
from pathlib import Path
import pwd
import shutil
import socket
import stat
import subprocess


def run(*args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def read_key(path):
    key = path.read_text().strip()
    if "\n" in key or not key.startswith("ssh-ed25519 "):
        raise ValueError(f"Expected one Ed25519 public key in {path}")
    run("ssh-keygen", "-lf", str(path), stdout=subprocess.DEVNULL)
    return key


def add_key(path, key, account):
    existing = path.read_text() if path.exists() else ""
    blob = key.split()[1]
    if not any(blob in line.split() for line in existing.splitlines()):
        with path.open("a") as output:
            if existing and not existing.endswith("\n"):
                output.write("\n")
            output.write(key + "\n")
    path.chmod(0o600)
    os.chown(path, account.pw_uid, account.pw_gid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-key", type=Path, required=True)
    parser.add_argument("--validation-public-key", type=Path)
    args = parser.parse_args()
    host = socket.gethostname().split(".")[0]
    if os.geteuid() != 0 or host not in ("amodo-gigabyte-3", "amodo-gigabyte-5"):
        raise SystemExit("Run as root on one of the two prepared instrument hosts")
    keys = [read_key(args.public_key)]
    if args.validation_public_key:
        keys.append(read_key(args.validation_public_key))
    users = [f"participant-{number}" for number in range(1, 21)]

    # Check every intended name and UID before making any account changes.
    for number, name in enumerate(users, start=1):
        try:
            account = pwd.getpwnam(name)
        except KeyError:
            try:
                conflict = pwd.getpwuid(2100 + number)
            except KeyError:
                continue
            raise RuntimeError(f"UID {2100 + number} belongs to {conflict.pw_name}")
        if account.pw_uid != 2100 + number or account.pw_dir != f"/home/{name}":
            raise RuntimeError(f"Existing account {name} has unexpected UID or home")

    if not shutil.which("setfacl"):
        run("apt-get", "install", "-y", "--no-install-recommends", "acl",
            env={**os.environ, "DEBIAN_FRONTEND": "noninteractive"})
    try:
        grp.getgrnam("aisb-participants")
    except KeyError:
        run("groupadd", "--system", "aisb-participants")

    # Traverse the administrator's home without granting directory listing or
    # access to its private files. Shared dependencies already have read access.
    run("setfacl", "-m", "g:aisb-participants:--x", "/home/tenant")

    # A root-owned lock is writable across users even with Linux's protected
    # regular-file rules for /tmp. Keep its inode to preserve existing flock use.
    lock = Path("/tmp/pico-workshop-capture.lock")
    if lock.is_symlink() or (lock.exists() and not stat.S_ISREG(lock.lstat().st_mode)):
        raise RuntimeError("Scope lock must be a regular file")
    lock.touch(exist_ok=True)
    os.chown(lock, 0, grp.getgrnam("users").gr_gid)
    lock.chmod(0o660)
    Path("/etc/tmpfiles.d/aisb-scope-lock.conf").write_text(
        "f /tmp/pico-workshop-capture.lock 0660 root users -\n"
        "x /tmp/pico-workshop-capture.lock - - - -\n"
    )

    # Copy only repository files, including uncommitted instructional changes.
    # Git's ignore rules exclude local credentials, raw captures, and environments.
    source = Path("/home/tenant/aisb")
    files = run("runuser", "-u", "tenant", "--", "git", "-C", str(source),
                "ls-files", "--cached", "--others", "--exclude-standard", "-z",
                capture_output=True).stdout
    selected = []
    for raw in files.split(b"\0"):
        if not raw:
            continue
        relative = Path(os.fsdecode(raw))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Unexpected repository path")
        if any(part in (".credentials", "instructor_setup", ".env", ".ssh")
               for part in relative.parts) or ".~" in relative.name:
            continue
        if (source / relative).is_file() or (source / relative).is_symlink():
            selected.append(raw)
    template = Path("/opt/aisb/participant-template-20261009")
    template.mkdir(parents=True, exist_ok=True)
    run("rsync", "-a", "--from0", "--files-from=-", str(source) + "/",
        str(template) + "/", input=b"\0".join(selected) + b"\0")
    run("chmod", "-R", "u+rwX,go+rX,go-w", str(template))

    result = []
    for number, name in enumerate(users, start=1):
        try:
            account = pwd.getpwnam(name)
        except KeyError:
            run("useradd", "--uid", str(2100 + number), "--user-group",
                "--create-home", "--home-dir", f"/home/{name}",
                "--shell", "/bin/bash", "--groups", "users,aisb-participants", name)
            account = pwd.getpwnam(name)
        run("usermod", "--append", "--groups", "users,aisb-participants", name)
        home = Path(account.pw_dir)
        home.chmod(0o700)
        ssh = home / ".ssh"
        ssh.mkdir(mode=0o700, exist_ok=True)
        ssh.chmod(0o700)
        os.chown(ssh, account.pw_uid, account.pw_gid)
        for key in keys:
            add_key(ssh / "authorized_keys", key, account)
        workspace = home / "aisb"
        if not workspace.exists():
            run("cp", "-a", "--reflink=auto", str(template), str(workspace))
            run("chown", "-R", f"{account.pw_uid}:{account.pw_gid}", str(workspace))
        run("sacctmgr", "-i", "add", "user", name, "Account=workshop",
            f"Cluster={host.replace('-', '_')}", "DefaultAccount=workshop",
            stdout=subprocess.DEVNULL)
        result.append({"user": name, "uid": account.pw_uid, "home": str(home),
                       "workspace": str(workspace), "slurm_account": "workshop"})
        print(f"Provisioned {name}", flush=True)
    report = Path("/var/lib/aisb/participants-20261009.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"host": host, "participants": result}, indent=2) + "\n")
    print(f"Saved {report}", flush=True)


if __name__ == "__main__":
    main()
