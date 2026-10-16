import csv
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for length in (1024, 4096):
    for repeat in (1, 2):
        data = json.loads((root / f'{length}_r{repeat}/round_summary.json').read_text())
        if not data['valid']:
            raise SystemExit('Cannot average invalid rounds')
        metrics = data['aisbench_metrics']
        def value(key):
            return float(str(metrics[key]['total']).split()[0])
        rows.append(dict(input_tokens=length, round=repeat, requests=data['requests'],
                         failed=data['failed'], total_tok_s=value('Total Token Throughput'),
                         output_tok_s=value('Output Token Throughput'),
                         duration_ms=value('Benchmark Duration')))
with (root / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
means = {str(length): {key: statistics.mean(row[key] for row in rows if row['input_tokens'] == length)
                       for key in ('total_tok_s', 'output_tok_s')}
         for length in (1024, 4096)}
(root / 'summary.json').write_text(json.dumps(dict(rounds=rows, means=means), indent=2), encoding='utf-8')
print('===== Mean of two valid rounds (tok/s) =====')
print(json.dumps(means, indent=2))
print(f'CSV: {root}/summary.csv')
