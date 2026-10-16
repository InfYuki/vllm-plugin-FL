"""Read-only endpoint/config validation; this file sends no inference requests."""
import ast
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1])
base = f"http://{os.environ['HOST_IP']}:{int(os.environ['PORT'])}"
headers = {}
if os.getenv('API_KEY'):
    headers['Authorization'] = 'Bearer ' + os.environ['API_KEY']
# Direct LAN access, independent of HTTP_PROXY settings inherited by the container.
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    request = urllib.request.Request(base + '/v1/models', headers=headers)
    with opener.open(request, timeout=15) as response:
        models = json.load(response)
except (OSError, urllib.error.URLError) as error:
    raise SystemExit(f'Cannot reach {base}/v1/models: {error}')
names = [item['id'] for item in models.get('data', [])]
if os.environ['MODEL_NAME'] not in names:
    raise SystemExit(f"MODEL_NAME={os.environ['MODEL_NAME']!r} not found; available: {names}")
config = out / 'config.py'
ast.parse(config.read_text(encoding='utf-8'))
meta = json.loads(config.with_suffix('.json').read_text(encoding='utf-8'))
model = meta['model']
assert model['host_ip'] == os.environ['HOST_IP']
assert model['host_port'] == int(os.environ['PORT'])
assert model['model'] == os.environ['MODEL_NAME']
assert model['retry'] >= 1
assert model['generation_kwargs']['ignore_eos'] is False
dataset = Path(meta['dataset']['path']) / 'test.jsonl'
with dataset.open(encoding='utf-8') as stream:
    count = sum(bool(line.strip()) for line in stream)
if count != 1319:
    raise SystemExit(f'Expected 1319 GSM8K test questions, found {count}')
if not Path(model['path']).is_dir():
    raise SystemExit(f"Local tokenizer/model path missing: {model['path']}")
record = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
              endpoint=base, model=os.environ['MODEL_NAME'], models_response=models,
              requested_questions=int(os.environ['ACC_NUM']), dataset_questions=count,
              dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
              aisbench_version=meta['aisbench_version'],
              note='Remote server flags and source version are not exposed by /v1/models.')
(out / 'remote_service.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print(f'Ready: {base}, model={os.environ["MODEL_NAME"]}; GSM8K data verified')
