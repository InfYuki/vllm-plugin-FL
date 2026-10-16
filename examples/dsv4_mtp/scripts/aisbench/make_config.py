"""Write literal AISBench configs; never instantiate an API client here."""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
mode, output = sys.argv[1:]
def integer(key, default):
    value = int(os.getenv(key, default))
    if value < 1:
        raise ValueError(f'{key} must be positive')
    return value

accuracy = mode == 'accuracy'
if mode not in ('accuracy', 'performance'):
    raise ValueError(mode)
model_type = 'VLLMCustomAPIChat' if accuracy else 'VLLMCustomAPI'
model = dict(attr='service', type=model_type, abbr='dsv4',
             path=os.environ['MODEL_PATH'], model=os.environ['MODEL_NAME'],
             # AISBench 3.1 uses range(retry): 1 = one attempt, no retry.
             stream=True, request_rate=0, use_timestamp=False, retry=1,
             api_key=os.getenv('API_KEY', ''), host_ip=os.environ['HOST_IP'],
             host_port=integer('PORT', 1112), url='', trust_remote_code=False,
             batch_size=integer('ACC_CC', 8) if accuracy else integer('CC', 64),
             max_out_len=integer('ACC_MAX_OUT', 8192) if accuracy else integer('OUTPUT_LEN', 1024),
             generation_kwargs=dict(temperature=0.0, top_p=1.0,
                                    seed=int(os.getenv('SEED', '1024')),
                                    ignore_eos=not accuracy))
infer = dict(prompt_template=dict(type='PromptTemplate', template='{question}'),
             retriever=dict(type='ZeroRetriever'), inferencer=dict(type='GenInferencer'))
if accuracy:
    if not 1 <= integer('ACC_NUM', 1319) <= 1319:
        raise ValueError('ACC_NUM must be 1..1319')
    infer['prompt_template']['template'] = dict(round=[dict(
        role='HUMAN', prompt='Solve the following math problem step by step. '
        'End your response with "answer: NUMBER", where NUMBER is the numerical answer.\n\n'
        'Question: {question}')])
    model['pred_postprocessor'] = dict(type='extract_non_reasoning_content')
    dataset = dict(abbr='gsm8k-0shot-cot', type='GSM8KDataset',
                   path=str(HERE / 'data/gsm8k'),
                   reader_cfg=dict(input_columns=['question'], output_column='answer'),
                   infer_cfg=infer,
                   eval_cfg=dict(evaluator=dict(type='Gsm8kEvaluator'),
                                 pred_postprocessor=dict(type='gsm8k_postprocess'),
                                 dataset_postprocessor=dict(type='gsm8k_dataset_postprocess')))
else:
    length = integer('INPUT_LEN', 1024)
    data = HERE / f'data/perf/seed{integer("SEED", 1024)}-n{integer("NUM_REQUESTS", 256)}-{length}.jsonl'
    if not data.is_file():
        raise FileNotFoundError(data)
    dataset = dict(abbr=f'fixed-{length}', type='CustomDataset', path=str(data),
                   reader_cfg=dict(input_columns=['question'], output_column='answer'),
                   infer_cfg=infer, eval_cfg=dict(evaluator=dict(type='AccEvaluator')))

# Imports are explicit so mmengine resolves registered types without remote code.
imports = '''from ais_bench.benchmark.models import VLLMCustomAPI, VLLMCustomAPIChat
from ais_bench.benchmark.datasets import CustomDataset, GSM8KDataset, Gsm8kEvaluator, gsm8k_postprocess, gsm8k_dataset_postprocess
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content
'''
import pprint
text = imports + '\nmodels = ' + pprint.pformat([model], sort_dicts=False)
text += '\ndatasets = ' + pprint.pformat([dataset], sort_dicts=False) + '\n'
text += '\nsummarizer = ' + repr(dict(attr='accuracy' if accuracy else 'performance')) + '\n'
import re
text = re.sub(r"('type': )'([A-Za-z_][A-Za-z_0-9]*)'", r'\1\2', text)
Path(output).write_text(text, encoding='utf-8')
meta = dict(mode=mode, model=model, dataset=dataset,
            aisbench_version=__import__('importlib.metadata', fromlist=['version']).version('ais-bench-benchmark'))
if not accuracy:
    import hashlib
    meta['dataset_sha256'] = hashlib.sha256(data.read_bytes()).hexdigest()
else:
    import hashlib
    meta['dataset_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (HERE / 'data/gsm8k').glob('*.jsonl')}
Path(output).with_suffix('.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
print(f'Config: {output}')
