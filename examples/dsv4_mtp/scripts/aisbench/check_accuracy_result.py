"""Require all requested GSM8K predictions and an evaluated accuracy score."""
import json
import sys
from pathlib import Path

root, expected = Path(sys.argv[1]), int(sys.argv[2])
predictions = [p for p in (root / 'work').rglob('gsm8k-0shot-cot.jsonl') if 'predictions' in p.parts]
evaluations = [p for p in (root / 'work').rglob('gsm8k-0shot-cot.json') if 'results' in p.parts]
rows = [json.loads(line) for p in predictions for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
failed_files = [p for p in (root / 'work').rglob('*_failed.jsonl') if p.stat().st_size]
score = json.loads(evaluations[0].read_text(encoding='utf-8')) if len(evaluations) == 1 else {}
valid = len(predictions) == 1 and len(rows) == expected and not failed_files and 'accuracy' in score
report = dict(valid=valid, expected_questions=expected, predictions=len(rows),
              accuracy=score.get('accuracy'), failed_files=[str(p) for p in failed_files],
              evaluation_files=[str(p) for p in evaluations])
(root / 'accuracy_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
if not valid:
    raise SystemExit('Incomplete GSM8K evaluation; inspect AISBench logs and failed requests')
