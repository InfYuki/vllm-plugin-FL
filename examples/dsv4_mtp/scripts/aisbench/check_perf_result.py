"""Reject incomplete/failed or wrong-length rounds and retain AISBench metrics."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
count, input_len, output_len = map(int, sys.argv[2:])
files = list((root / 'work').rglob(f'fixed-{input_len}_details.jsonl'))
if len(files) != 1:
    raise SystemExit(f'Expected one AISBench request detail file, found {len(files)}; inspect {root}/console.log')
rows = [json.loads(line) for line in files[0].read_text(encoding='utf-8').splitlines() if line.strip()]
failed = sum(not row.get('success') for row in rows)
wrong_input = sum(row.get('input_tokens') != input_len for row in rows)
wrong_output = sum(row.get('output_tokens') != output_len for row in rows)
summary = dict(requests=len(rows), expected_requests=count, failed=failed,
               wrong_input_length=wrong_input, wrong_output_length=wrong_output,
               request_details=str(files[0]))
metrics = list((root / 'work').rglob(f'fixed-{input_len}.json'))
metrics = [p for p in metrics if 'performances' in p.parts]
if len(metrics) == 1:
    summary['aisbench_metrics'] = json.loads(metrics[0].read_text(encoding='utf-8'))
summary['valid'] = len(rows) == count and not (failed or wrong_input or wrong_output) and len(metrics) == 1
(root / 'round_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(summary, indent=2, ensure_ascii=False))
if not summary['valid']:
    raise SystemExit('Invalid round: missing/failed requests, wrong token counts or missing metrics; see round_summary.json')
