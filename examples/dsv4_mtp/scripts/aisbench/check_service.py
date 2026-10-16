"""GET readiness and record existing launch evidence. Never generate or start services."""
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

variant, output = sys.argv[1:]
output = Path(output)
experiment = Path(__file__).resolve().parent.parent
launch = experiment / 'runs' / variant / 'launch_metadata.json'
if not launch.is_file():
    raise SystemExit(f'Missing {launch}. Start the requested {variant} service yourself first.')
meta = json.loads(launch.read_text())
if meta['variant'] != variant or int(meta['port']) != int(os.environ['PORT']):
    raise SystemExit('Requested variant/PORT differs from launch metadata')
stat = Path(f'/proc/{meta["pid"]}/stat')
if not stat.exists() or stat.read_text().rsplit(')', 1)[1].split()[19] != str(meta['starttime']):
    raise SystemExit('Recorded server PID is no longer alive. Start the service yourself first.')
manifest = experiment / 'manifest.json'
if hashlib.sha256(manifest.read_bytes()).hexdigest() != meta['manifest_sha256']:
    raise SystemExit('Service was started with a previous source manifest. Restart it yourself before comparing.')
for src in (launch, manifest):
    (output / src.name).write_bytes(src.read_bytes())
url = f'http://{os.environ["HOST_IP"]}:{os.environ["PORT"]}/v1/models'
deadline = time.monotonic() + int(os.getenv('WAIT_SECONDS', '300'))
headers = {'Authorization': 'Bearer ' + os.environ['API_KEY']} if os.getenv('API_KEY') else {}
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
print(f'Waiting for {variant}: {url}', flush=True)
while True:
    try:
        with opener.open(urllib.request.Request(url, headers=headers), timeout=5) as response:
            models = json.load(response)
        if os.environ['MODEL_NAME'] not in [m['id'] for m in models.get('data', [])]:
            raise RuntimeError('MODEL_NAME is absent from /v1/models')
        (output / 'service_models.json').write_text(json.dumps(models, indent=2), encoding='utf-8')
        print(f'Service ready: {variant}; graph={meta.get("graph")}; async={meta.get("async_scheduling")}', flush=True)
        break
    except Exception as error:
        if time.monotonic() >= deadline:
            raise SystemExit(f'Service not ready: {error}; inspect {experiment}/runs/{variant}/server.log')
        print(f'Not ready: {error}; retry in 5 seconds', flush=True)
        time.sleep(5)
