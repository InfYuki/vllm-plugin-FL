"""Read-only identity/hash checks. Does not initialize torch or the NPU."""
import hashlib,importlib.metadata as md,importlib.util,json,os,pathlib,subprocess,urllib.parse
here=pathlib.Path(__file__).resolve().parent
m=json.loads((here/'manifest.json').read_text()); repo=pathlib.Path(m['repo'])
git=lambda *args:subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
assert git('rev-parse','HEAD')==m['commit'], 'FL commit changed'
for r in m['preserved']+m['changed']:
    assert hashlib.sha256((repo/r['path']).read_bytes()).hexdigest()==r['sha256'], r['path']+' hash mismatch'
expected=set() if m.get('committed_source') else {r['path'] for r in m['preserved']+m['changed'] if r.get('before_sha256','existing') is not None}
assert set(git('diff','HEAD','--name-only').splitlines())==expected, 'Unexpected tracked source changes'
added={r['path'] for r in m['changed'] if r['before_sha256'] is None}
untracked=git('ls-files','--others','--exclude-standard','--','vllm_fl').splitlines()
assert all(p in added or p.endswith('.so') or p.startswith('vllm_fl/_cann_ops_custom/') for p in untracked), 'Unexpected package files'
assert all(v in ('','0') for n,v in os.environ.items() if n.startswith('FL_DSV4_')), 'Disable old performance patches for this experiment'
dist=md.distribution('vllm-plugin-fl'); direct=json.loads(dist.read_text('direct_url.json'))
assert direct.get('dir_info',{}).get('editable')
assert pathlib.Path(urllib.parse.unquote(urllib.parse.urlparse(direct['url']).path)).resolve()==repo.resolve()
assert pathlib.Path(importlib.util.find_spec('vllm_fl').origin).resolve().parent==repo/'vllm_fl'
assert list((repo/'vllm_fl').glob('_C_ascend*.so')) and (repo/'vllm_fl/libvllm_fl_ascend_kernels.so').is_file()
print(json.dumps(dict(experiment=m['experiment'],version=dist.version,commit=m['commit'],source_sha256=hashlib.sha256((here/'manifest.json').read_bytes()).hexdigest()),indent=2))
