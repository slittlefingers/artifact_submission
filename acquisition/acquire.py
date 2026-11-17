#!/usr/bin/env python3
"""Acquire physical memory from a rooted Android device over USB with LEMON.

Steps:
  1. Select the adb device.
  2. Check root access through su.
  3. Make sure the LEMON binary is in the device's temporary directory;
     push it if it is missing or differs from the local copy.
  4. Export kernel symbols from the current boot: the BTF blob
     (/sys/kernel/btf/vmlinux) and /proc/kallsyms, with kptr_restrict
     temporarily set to 0. Symbols must come from the same boot as the dump.
  5. Check that USB tethering is up and find the host address on that link.
  6. Ask for confirmation, then stream the dump over the USB network link
     into a local LiME image. Nothing is written to device storage.

Only the Python standard library and adb are required.
"""
import argparse
import datetime as dt
import hashlib
import ipaddress
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
USB_IFACE = re.compile(r'^(ncm|rndis|usb)\d+$')
LIME_MAGIC = b'EMiL'                                     
BTF_MAGIC = b'\x9f\xeb'                              
CHUNK = 1 << 20


def fail(msg):
    sys.exit(f'[!] {msg}')


def info(msg):
    print(f'[*] {msg}', flush=True)


class Device:
    def __init__(self, adb, serial):
        self.base = [adb] + (['-s', serial] if serial else [])
        self.su = None

    def run(self, args, timeout=60):
        r = subprocess.run(self.base + args, capture_output=True, timeout=timeout)
        return r.returncode, r.stdout.decode('utf-8', 'replace').replace('\r\n', '\n')

    def shell(self, command, timeout=60):
        return self.run(['shell', command], timeout)[1]

    def root(self, command, timeout=60):
        return self.shell(f'{self.su} -c {shlex.quote(command)}', timeout)

    def root_bytes(self, command, timeout=300):
        cmd = self.base + ['exec-out', f'{self.su} -c {shlex.quote(command)}']
        return subprocess.run(cmd, capture_output=True, timeout=timeout).stdout


def select_device(adb, serial):
    try:
        out = subprocess.run([adb, 'devices'], capture_output=True, text=True,
                             timeout=30).stdout
    except FileNotFoundError:
        fail(f'adb not found ({adb}). Install platform-tools or pass --adb.')
    ready = [l.split('\t')[0] for l in out.splitlines()[1:] if l.endswith('\tdevice')]
    if serial:
        if serial not in ready:
            fail(f'Device {serial} is not connected or not authorized.')
        return serial
    if len(ready) != 1:
        fail(f'Expected one connected device, found {len(ready)}. Use --serial.')
    return ready[0]


def check_root(dev, candidates):
    for su in candidates:
        dev.su = su
        try:
            if 'uid=0' in dev.root('id', timeout=30):
                info(f'Root access confirmed via "{su}".')
                return
        except subprocess.TimeoutExpired:
            pass
    fail('No root access. Grant su to the adb shell, or pass --su with the su path.')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(CHUNK), b''):
            h.update(block)
    return h.hexdigest()


def remote_digest(dev, remote):
    out = dev.shell(f'sha256sum {remote} 2>/dev/null').split()
    return out[0] if out else None


def ensure_lemon(dev, local, remote_dir):
    remote = f'{remote_dir}/{local.name}'
    want = sha256_file(local)
    if remote_digest(dev, remote) == want:
        info(f'LEMON already present on the device: {remote}')
        return remote
    info(f'LEMON missing or outdated on the device; pushing {local.name} ...')
    code, out = dev.run(['push', str(local), remote], timeout=300)
    if code != 0:
        fail(f'adb push failed:\n{out}')
    dev.shell(f'chmod 755 {remote}')
    have = remote_digest(dev, remote)
    if have is None:
        size = dev.shell(f'stat -c %s {remote} 2>/dev/null').strip()
        if size != str(local.stat().st_size):
            fail('Pushed LEMON binary does not match the local copy.')
        info(f'LEMON pushed (size verified; sha256sum unavailable): {remote}')
    elif have != want:
        fail('Pushed LEMON binary does not match the local copy.')
    else:
        info(f'LEMON pushed and verified: {remote}')
    return remote


def export_symbols(dev, btf_path, kallsyms_path):
    original = dev.root('cat /proc/sys/kernel/kptr_restrict').strip()
    dev.root('echo 0 > /proc/sys/kernel/kptr_restrict')
    if dev.root('cat /proc/sys/kernel/kptr_restrict').strip() != '0':
        print('[!] Could not set kptr_restrict to 0; kallsyms addresses may be hidden.')
    try:
        kallsyms = dev.root_bytes('cat /proc/kallsyms')
    finally:
        if original and original != '0':
            dev.root(f'echo {original} > /proc/sys/kernel/kptr_restrict')
            info(f'kptr_restrict restored to {original}.')
    lines = kallsyms.splitlines()
    hidden = sum(1 for l in lines if l[:16].strip(b'0') == b'')
    if not lines or hidden > len(lines) // 2:
        fail('kallsyms is empty or its addresses are hidden; check kptr_restrict.')
    kallsyms_path.write_bytes(kallsyms)
    info(f'Saved kallsyms ({len(lines):,} symbols): {kallsyms_path}')

    btf = dev.root_bytes('cat /sys/kernel/btf/vmlinux')
    if btf[:2] != BTF_MAGIC:
        print('[!] /sys/kernel/btf/vmlinux is missing or invalid; BTF not saved.')
        return
    btf_path.write_bytes(btf)
    info(f'Saved BTF ({len(btf):,} bytes): {btf_path}')


def usb_link(dev, host_ip=None):
    links = []
    for line in dev.root('ip -4 -o addr show').splitlines():
        m = re.match(r'^\d+:\s+(\S+)\s+inet\s+(\S+)', line)
        if m and USB_IFACE.match(m.group(1)):
            links.append((m.group(1), ipaddress.ip_interface(m.group(2))))
    if not links:
        fail('USB tethering is off. Enable it on the device (Settings > Network & '
             'internet > Hotspot & tethering > USB tethering) and retry.')
    for name, iface in links:
        if host_ip:
            host = ipaddress.ip_address(host_ip)
        else:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((str(iface.ip), 9))                         
                host = ipaddress.ip_address(s.getsockname()[0])
        if host in iface.network:
            return name, iface, host
    fail('USB tethering is on, but the host has no address on the USB network yet. '
         'Wait for the host adapter to obtain an address, or pass --host-ip.')


class Receiver(threading.Thread):
    def __init__(self, listener, out_path, accept_timeout):
        super().__init__(daemon=True)
        self.listener, self.out_path = listener, out_path
        self.accept_timeout = accept_timeout
        self.connected, self.stop = threading.Event(), threading.Event()
        self.total, self.sha256, self.error, self.peer = 0, hashlib.sha256(), None, None
        self.started = None

    def run(self):
        deadline = time.monotonic() + self.accept_timeout
        self.listener.settimeout(1.0)
        while True:
            try:
                conn, self.peer = self.listener.accept()
                break
            except socket.timeout:
                if self.stop.is_set() or time.monotonic() > deadline:
                    self.error = ('LEMON did not connect to the host. If the log shows a '
                                  'connection error, allow inbound TCP on this port in '
                                  'the host firewall.')
                    return
            except OSError as e:
                self.error = f'Listener failed: {e}'
                return
        conn.settimeout(None)
        self.connected.set()
        self.started = last = time.monotonic()
        buf = bytearray(CHUNK)
        view = memoryview(buf)
        try:
            with conn, open(self.out_path, 'xb') as f:
                while True:
                    n = conn.recv_into(view)
                    if n == 0:
                        break
                    f.write(view[:n])
                    self.sha256.update(view[:n])
                    self.total += n
                    now = time.monotonic()
                    if now - last >= 1:
                        rate = self.total / (now - self.started) / 2**20
                        print(f'\r    {self.total / 2**20:,.0f} MiB  {rate:,.1f} MiB/s',
                              end='', flush=True)
                        last = now
        except OSError as e:
            self.error = f'Transfer interrupted: {e}'
        print()


def acquire(dev, remote_lemon, host, port, out_path, huge, raw, accept_timeout):
    log_path = out_path.with_name(out_path.name + '.lemon.log')
    try:
        listener = socket.create_server((str(host), port))
    except OSError as e:
        fail(f'Cannot listen on {host}:{port} ({e}). Is another receiver running?')
    receiver = Receiver(listener, out_path, accept_timeout)
    receiver.start()
    cmd = f'{remote_lemon} -n {host} -p {port}' + (' -H' if huge else '') + (' -w' if raw else '')
    info(f'Listening on {host}:{port}; starting LEMON: {cmd}')
    with open(log_path, 'wb') as log:
        proc = subprocess.Popen(dev.base + ['shell', f'{dev.su} -c {shlex.quote(cmd)}'],
                                stdout=log, stderr=subprocess.STDOUT)
        try:
            while receiver.is_alive():
                receiver.join(0.5)
                if proc.poll() is not None and not receiver.connected.is_set():
                    receiver.stop.set()                                       
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            receiver.stop.set()
            fail(f'Interrupted. Partial image (if any): {out_path}')
        finally:
            listener.close()

    if receiver.error or receiver.total == 0:
        tail = log_path.read_text('utf-8', 'replace').strip().splitlines()[-15:]
        fail((receiver.error or 'No data received.') + f'\nLEMON log ({log_path}):\n'
             + '\n'.join(tail))
    with open(out_path, 'rb') as f:
        head = f.read(4)
    if not raw and head != LIME_MAGIC:
        print('[!] The image does not start with a LiME header.')
    if proc.returncode != 0:
        print(f'[!] LEMON exited with status {proc.returncode}; see {log_path}.')
    digest = receiver.sha256.hexdigest()
    out_path.with_name(out_path.name + '.sha256').write_text(f'{digest}  {out_path.name}\n')
    secs = time.monotonic() - receiver.started
    info(f'Done: {receiver.total:,} bytes in {secs:,.0f} s -> {out_path}')
    info(f'SHA-256: {digest}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--adb', default=os.environ.get('ADB', 'adb'), help='adb executable')
    ap.add_argument('--serial', help='adb serial when several devices are connected')
    ap.add_argument('--su', action='append',
                    help='su command to try (repeatable; default: su, /data/local/tmp/su)')
    ap.add_argument('--lemon', type=Path, default=HERE / 'lemon.core.aarch64',
                    help='local LEMON binary (use lemon.nocoreuni.aarch64 if the '
                         'device kernel has no BTF)')
    ap.add_argument('--remote-dir', default='/data/local/tmp', help='device directory for LEMON')
    ap.add_argument('--out', type=Path, help='output image (default: mem_<time>.lime)')
    ap.add_argument('--port', type=int, default=2304, help='TCP port on the host')
    ap.add_argument('--host-ip', help='host address on the USB network (default: auto)')
    ap.add_argument('--raw', action='store_true', help='RAW image instead of LiME (-w)')
    ap.add_argument('--no-huge', action='store_true', help='do not pass -H (2 MB pages)')
    ap.add_argument('--skip-symbols', action='store_true', help='do not export BTF/kallsyms')
    ap.add_argument('--accept-timeout', type=float, default=120,
                    help='seconds to wait for LEMON to connect')
    ap.add_argument('--yes', action='store_true', help='do not ask for confirmation')
    args = ap.parse_args()

    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = args.out or Path(f'mem_{stamp}' + ('.raw' if args.raw else '.lime'))
    out = out.resolve()
    if out.exists():
        fail(f'Output already exists: {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    if not args.lemon.is_file():
        fail(f'LEMON binary not found: {args.lemon}')

    serial = select_device(args.adb, args.serial)
    info(f'Device: {serial}')
    dev = Device(args.adb, serial)
    check_root(dev, args.su or ['su', '/data/local/tmp/su'])
    remote_lemon = ensure_lemon(dev, args.lemon, args.remote_dir)

    if not args.skip_symbols:
        export_symbols(dev, out.with_name(out.stem + '.vmlinux.btf'),
                       out.with_name(out.stem + '.kallsyms.txt'))

    name, iface, host = usb_link(dev, args.host_ip)
    info(f'USB tethering is up: device {iface.ip} ({name}) <-> host {host}')

    mem_kib = re.search(r'MemTotal:\s+(\d+)', dev.shell('cat /proc/meminfo'))
    if mem_kib:
        ram = int(mem_kib.group(1)) * 1024
        free = shutil.disk_usage(out.parent).free
        info(f'Device RAM {ram / 2**30:.1f} GiB; free space at {out.parent}: '
             f'{free / 2**30:.1f} GiB')
        if free < ram * 1.25:                                                    
            print('[!] The output disk may not have enough free space for a full image.')

    if not args.yes:
        reply = input('Start memory acquisition now? [y/N] ').strip().lower()
        if reply not in ('y', 'yes'):
            info('Aborted by user; nothing was acquired.')
            return
    acquire(dev, remote_lemon, host, args.port, out, not args.no_huge, args.raw,
            args.accept_timeout)


if __name__ == '__main__':
    main()
