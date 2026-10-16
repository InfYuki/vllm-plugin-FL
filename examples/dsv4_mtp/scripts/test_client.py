"""Client only: record/compare greedy token IDs, then benchmark explicitly."""
import argparse,concurrent.futures,hashlib,json,os,pathlib,subprocess,time,urllib.request
p=argparse.ArgumentParser()
p.add_argument('variant',choices=['baseline','mtp1','mtp2','mtp3'])
p.add_argument('mode',choices=['correctness','performance'])
p.add_argument('--concurrency',type=int,nargs='+',default=[1,12,64])
a=p.parse_args(); here=pathlib.Path(__file__).resolve().parent
manifest=json.loads((here/'manifest.json').read_text())
meta=json.loads((here/'runs'/a.variant/'launch_metadata.json').read_text())
assert meta['experiment']==manifest['experiment'] and meta['variant']==a.variant
assert meta['manifest_sha256']==hashlib.sha256((here/'manifest.json').read_bytes()).hexdigest()
assert meta['port']==int(os.getenv('PORT','1112'))
stat=pathlib.Path(f"/proc/{meta['pid']}/stat").read_text().rsplit(')',1)[1].split()
assert stat[19]==meta['starttime'] and stat[0]!='Z', 'Recorded service has exited'
server_env=dict(x.split(b'=',1) for x in pathlib.Path(f"/proc/{meta['pid']}/environ").read_bytes().split(b'\0') if b'=' in x)
assert server_env.get(b'VLLM_PLUGINS')==b'fl'
assert all(v in (b'',b'0') for n,v in server_env.items() if n.startswith(b'FL_DSV4_'))
subprocess.run(['python',str(here/'check_source.py')],check=True)
base=f"http://127.0.0.1:{meta['port']}"
out=here/'results'/a.variant/(time.strftime('%Y%m%d-%H%M%S')+'-'+a.mode)
out.mkdir(parents=True,exist_ok=False)
(out/'launch_metadata.json').write_text(json.dumps(meta,indent=2))
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
def api(path,payload=None):
    request=urllib.request.Request(base+path,data=json.dumps(payload).encode() if payload is not None else None,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=1800) as response:return json.load(response)
def metrics(tag):
    with urllib.request.urlopen(base+'/metrics',timeout=30) as response:text=response.read().decode()
    (out/(tag+'.prom')).write_text(text)
    totals={}
    for line in text.splitlines():
        if line.startswith('vllm:spec_decode_') and not line.startswith('#'):
            name=line.split('{',1)[0].split()[0]
            totals[name]=totals.get(name,0.)+float(line.rsplit(' ',1)[1])
    return totals
try:
    api('/v1/models')
except Exception as exc:
    raise SystemExit(f'Service is not ready at {base}; inspect {here}/runs/{a.variant}/server.log: {exc}')
before=metrics('before')
oracle=here/'oracles'/f"greedy-graph{meta['graph']}.json"
approval=here/'oracles'/f"{a.variant}-graph{meta['graph']}-passed.json"
if a.mode=='correctness':
    if a.variant=='baseline':
        seed=api('/tokenize',{'model':'dsv4','prompt':'The quick brown fox jumps over the lazy dog. Explain the pattern precisely. '})['tokens']
        lengths=[1,31,32,33,127,128,129,1024,4096,18000]
        cases=[dict(id=f'len{n}',prompt=(seed*((n+len(seed)-1)//len(seed)))[:n],max_tokens=96) for n in lengths]
        cases += [dict(id='normal-eos',prompt='Answer with one word: what is 2 plus 2?',max_tokens=96,ignore_eos=False)]
        reference=None
    else:
        reference=json.loads(oracle.read_text())
        assert reference['manifest_sha256']==meta['manifest_sha256'] and reference['graph']==meta['graph']
        assert reference['concurrency']==a.concurrency, 'Use the same concurrency set as the oracle'
        cases=reference['cases']
    records={}; mismatches=[]
    def request(item):
        index,case=item
        payload=dict(model='dsv4',prompt=case['prompt'],max_tokens=case['max_tokens'],temperature=0,seed=1024,ignore_eos=case.get('ignore_eos',True),return_token_ids=True)
        result=api('/v1/completions',payload)
        choice=result['choices'][0]; tokens=choice.get('token_ids')
        assert isinstance(tokens,list) and tokens, 'Server must return nonempty token_ids'
        if payload['ignore_eos']:assert len(tokens)==payload['max_tokens']
        return index,dict(token_ids=tokens,finish_reason=choice['finish_reason'],usage=result['usage'])
    for concurrency in a.concurrency:
        assert 1<=concurrency<=64
        # At least one full batch, followed by another batch to exercise reuse.
        inputs=[(f'cc{concurrency}-req{i}',cases[i%len(cases)]) for i in range(max(len(cases),concurrency*2))]
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            for key,value in pool.map(request,inputs):
                records[key]=value
                if reference is not None and value['token_ids']!=reference['records'][key]['token_ids']:
                    mismatches.append(key)
        print('completed concurrency',concurrency,flush=True)
    record=dict(cases=cases,records=records,graph=meta['graph'],concurrency=a.concurrency,manifest_sha256=meta['manifest_sha256'],launch_timestamp=meta['timestamp'],mismatches=mismatches)
    (out/'correctness.json').write_text(json.dumps(record,indent=2))
else:
    approved=json.loads(approval.read_text())
    assert approved['launch_timestamp']==meta['timestamp'], 'Run correctness for this service before benchmarking'
    assert approved['manifest_sha256']==meta['manifest_sha256']
    assert approved['concurrency']==[1,12,64], 'Complete the full correctness suite before benchmarking'
    def bench(length,tag,count=256,output=1024):
        command=['vllm','bench','serve','--backend','vllm','--model',manifest['model'],'--served-model-name','dsv4','--tokenizer',manifest['model'],'--tokenizer-mode','deepseek_v4','--endpoint','/v1/completions','--host','127.0.0.1','--port',str(meta['port']),'--dataset-name','random','--temperature','0','--random-input-len',str(length),'--random-output-len',str(output),'--num-prompts',str(count),'--max-concurrency','64','--ignore-eos','--seed','1024','--save-result','--result-dir',str(out),'--result-filename',tag+'.json']
        (out/(tag+'.command.json')).write_text(json.dumps(command,indent=2))
        with (out/(tag+'.log')).open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
        result=json.loads((out/(tag+'.json')).read_text())
        assert result['completed']==count and result.get('failed',0)==0
        assert isinstance(result.get('total_token_throughput'),(int,float))
        return result
    bench(1024,'warmup',64,128)
    summary={}
    for length in [1024,4096]:
        summary[str(length)]=[bench(length,f'{length}-repeat{i}')['total_token_throughput'] for i in [1,2]]
    (out/'throughput.json').write_text(json.dumps(summary,indent=2))
after=metrics('after')
delta={k:after.get(k,0)-before.get(k,0) for k in set(after)|set(before)}
(out/'spec_metrics_delta.json').write_text(json.dumps(delta,indent=2))
if a.variant!='baseline':
    drafts=delta.get('vllm:spec_decode_num_draft_tokens_total',0)
    accepted=delta.get('vllm:spec_decode_num_accepted_tokens_total',0)
    assert drafts>0 and accepted>0, 'No accepted MTP drafts observed; inspect per-rank logs and weights'
if a.mode=='correctness':
    assert not mismatches, f'{len(mismatches)} token-ID mismatches; see {out}/correctness.json'
    oracle.parent.mkdir(exist_ok=True)
    if a.variant=='baseline':
        if oracle.exists():
            archive=oracle.with_name(oracle.stem+f'.{time.time_ns()}.json');oracle.replace(archive)
        oracle.write_text(json.dumps(record,indent=2))
    approval.write_text(json.dumps(record,indent=2))
print('PASS',out)
