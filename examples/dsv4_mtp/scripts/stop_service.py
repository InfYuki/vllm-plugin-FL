"""Stop only the PID/starttime recorded by this package and its descendants."""
import json
import os
import pathlib
import signal
import sys
import time

variant=sys.argv[1]
assert variant in ('baseline','mtp1','mtp2','mtp3')
here=pathlib.Path(__file__).resolve().parent
record=json.loads((here/'runs'/variant/'identity.json').read_text())

def info(pid):
    try:
        fields=pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return fields[19],int(fields[1]),fields[0]
    except (OSError,ValueError):
        return None

pid=record['pid'];root=info(pid)
if root is None or root[0]!=record['starttime'] or root[2]=='Z':
    raise SystemExit('Recorded server is no longer alive; no process killed.')
cmd=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes()
assert b'vllm' in cmd and b'DeepSeek-V4-Flash-w8a8-mtp' in cmd, 'PID command differs'
parents={};stamps={pid:root[0]}
for entry in pathlib.Path('/proc').glob('[0-9]*/stat'):
    child=int(entry.parent.name);data=info(child)
    if data: parents.setdefault(data[1],[]).append((child,data[0]))
queue=[pid]
while queue:
    parent=queue.pop()
    for child,stamp in parents.get(parent,[]):
        if child not in stamps: stamps[child]=stamp;queue.append(child)

def send(sig):
    for target,stamp in stamps.items():
        data=info(target)
        if data and data[0]==stamp and data[2]!='Z':
            try: os.kill(target,sig)
            except ProcessLookupError: pass

send(signal.SIGTERM)
for _ in range(30):
    if all((d:=info(p)) is None or d[0]!=s or d[2]=='Z' for p,s in stamps.items()): break
    time.sleep(1)
send(signal.SIGKILL)
print('Stopped only recorded service and identified descendants.')
