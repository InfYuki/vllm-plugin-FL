"""CPU-only generation of fixed-length text prompts. No service/API requests."""
import hashlib
import json
import os
import random
from pathlib import Path
from transformers import PreTrainedTokenizerFast

HERE = Path(__file__).resolve().parent
model = Path(os.environ['MODEL_PATH'])
seed = int(os.getenv('SEED', '1024'))
count = int(os.getenv('NUM_REQUESTS', '256'))
if count < 1 or seed < 1:
    raise ValueError('NUM_REQUESTS and SEED must be positive')
tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model), local_files_only=True)
special = set(tokenizer.all_special_ids)
vocab = [i for i in range(tokenizer.vocab_size) if i not in special]
fingerprint = hashlib.sha256((model / 'tokenizer.json').read_bytes()).hexdigest()
target_dir = HERE / 'data/perf'
target_dir.mkdir(parents=True, exist_ok=True)
for length in (1024, 4096):
    file = target_dir / f'seed{seed}-n{count}-{length}.jsonl'
    manifest = file.with_suffix('.meta.json')
    if file.exists() and manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved['tokenizer_sha256'] != fingerprint or saved['sha256'] != hashlib.sha256(file.read_bytes()).hexdigest():
            raise RuntimeError(f'Cached dataset changed: {file}; preserve results, remove cache and regenerate')
        print(f'Reusing {file} sha256={saved["sha256"]}', flush=True)
        continue
    rng = random.Random(seed + length)
    rows = []
    for index in range(count):
        # Random vocabulary text, retokenized and checked after decoding. A raw
        # decode of N random token IDs does not reliably remain N tokens.
        text = tokenizer.decode(rng.choices(vocab, k=length * 2), skip_special_tokens=True)
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if len(encoded['input_ids']) < length:
            raise RuntimeError('Insufficient prompt material')
        end = encoded['offset_mapping'][length - 1][1]
        prompt = None
        for cut in [end] + list(range(max(1, end - 24), min(len(text), end + 24) + 1)):
            candidate = text[:cut]
            if len(tokenizer.encode(candidate, add_special_tokens=False)) == length:
                prompt = candidate
                break
        if prompt is None:
            raise RuntimeError(f'Cannot form exactly {length} tokens for row {index}')
        rows.append(json.dumps(dict(question=prompt, answer=''), ensure_ascii=False))
    blob = ('\n'.join(rows) + '\n').encode('utf-8')
    temp = file.with_suffix('.tmp')
    temp.write_bytes(blob)
    temp.replace(file)
    saved = dict(seed=seed, requests=count, input_len=length,
                 tokenizer_sha256=fingerprint, sha256=hashlib.sha256(blob).hexdigest(),
                 generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    manifest.write_text(json.dumps(saved, indent=2), encoding='utf-8')
    print(f'Prepared {count} prompts of exactly {length} tokens: {file}', flush=True)
