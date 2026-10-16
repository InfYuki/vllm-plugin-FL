#!/usr/bin/env python3
"""Prepare a new launch/evaluation workspace. No install, service or evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', default='/workspace/model_test/dsv4-mtp-infyuki-20261015')
    parser.add_argument('--container', default='fl-infyuki-v024-hzj')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', args.container):
        parser.error('container must be a Docker container name')
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+', args.experiment_dir):
        parser.error('experiment-dir must be an absolute POSIX path without spaces')
    here = Path(__file__).resolve().parent
    repo = here.parents[1]
    target = Path(args.experiment_dir).resolve()
    if target.exists():
        raise SystemExit(f'Refusing to overwrite existing workspace: {target}; choose a new path')
    if target == repo or repo in target.parents:
        raise SystemExit('Put experiment output outside the source checkout')
    metadata = json.loads((here / 'source_snapshot.json').read_text())
    git = lambda *parts: subprocess.check_output(['git', '-C', str(repo), *parts], text=True).strip()
    if git('diff', 'HEAD', '--name-only'):
        raise SystemExit('Prepare from a clean committed source checkout')
    for row in metadata['changed'] + metadata['preserved']:
        if hashlib.sha256((repo / row['path']).read_bytes()).hexdigest() != row['sha256']:
            raise SystemExit('Source differs from published snapshot: ' + row['path'])
    runtime = Path('/vllm-workspace/vllm')
    runtime_head = subprocess.check_output(['git', '-C', str(runtime), 'rev-parse', 'HEAD'], text=True).strip()
    if runtime_head != metadata['runtime_vllm']:
        raise SystemExit(f'vLLM revision differs: {runtime_head}; required {metadata["runtime_vllm"]}')
    old_exp = '/workspace/model_test/dsv4-mtp-infyuki-20261015'
    shutil.copytree(here / 'scripts', target)
    for path in target.rglob('*'):
        if path.is_file() and path.suffix in ('.py', '.sh'):
            text = path.read_text().replace(old_exp, str(target))
            text = text.replace('fl-infyuki-v024-hzj', args.container)
            path.write_text(text, encoding='utf-8', newline='\n')
    metadata['source_base_commit'] = metadata['commit']
    metadata['commit'] = git('rev-parse', 'HEAD')
    metadata['repo'] = str(repo)
    metadata['committed_source'] = True
    metadata['migration_note'] = 'Prepared from committed snapshot; no installation, inference or validation performed'
    (target / 'manifest.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    for row in metadata['changed']:
        path = target / 'payload' / row['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / row['path'], path)
    (target / 'aisbench/data/gsm8k').mkdir(parents=True)
    (target / 'aisbench/data/perf').mkdir(parents=True)
    print(f'Prepared {target}')
    print('Copy the previous aisbench/data directory here before accuracy/performance comparison.')
    print('No dependencies installed, services started/stopped, tests or API requests run.')


if __name__ == '__main__':
    main()
