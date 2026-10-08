"""CPU-only embedding/projection diagnostics; no language-model forward or generation."""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import collections
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import resource
import struct
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'e01'))
from audit_e01 import ROOT, CONTRACT, atomic, digest, donors, inventory, jsonl, order, read, require, sha, stats, token_spans
from validate_e01 import validate as validate_parent_e01

PARENT = ROOT/'outputs/mab/review_m3_formal/20261002-m3-six-policy-v3'
CORE = ('queries.npz', 'query_metrics.jsonl', 'geometry.json', 'summary.json')


@dataclass
class ConfigProxy:
    """Data-only allowlist target; never instantiates a real memory bank."""


@dataclass
class SlotProxy:
    """Only attributes from a checksum-verified local snapshot are restored."""


def load_bank(path):
    allow = [(ConfigProxy, 'memgen.model.latent_memory_bank.LatentMemoryBankConfig'),
             (SlotProxy, 'memgen.model.latent_memory_bank.LatentMemorySlot'),
             np.ndarray, np.dtype, np._core.multiarray._reconstruct,
             type(np.dtype(np.uint32)), torch.torch_version.TorchVersion]
    with torch.serialization.safe_globals(allow):
        value = torch.load(path, map_location='cpu', weights_only=True)
    require(value['schema'] == 'm2-bank/v1', 'snapshot schema')
    slots = value['bank']['_slots']
    require(len(slots) == 16 and all(tuple(s.key.shape) == (1536,) for s in slots), 'key shape')
    return torch.stack([s.key for s in slots]), value['bank']['config']


def tensor_hash(t):
    t = t.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(t.dtype).encode('ascii'))
    h.update(json.dumps(list(t.shape)).encode('ascii'))
    h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class EmbeddingRows:
    def __init__(self, path):
        with Path(path).open('rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(n))
        info = header['model.embed_tokens.weight']
        require(info['dtype'] == 'BF16' and info['shape'] == [151936,1536], 'embedding metadata')
        lo, hi = info['data_offsets']
        require(hi-lo == 151936*1536*2, 'embedding storage')
        self.map = np.memmap(path, dtype='<u2', mode='r', offset=8+n+lo, shape=tuple(info['shape']))
        self.cache = {}
        self.metadata = info

    def get(self, ids):
        require(ids and all(0 <= i < self.map.shape[0] for i in ids), 'bad token IDs')
        for i in set(ids)-self.cache.keys():
            self.cache[i] = torch.from_numpy(self.map[i].copy()).view(torch.bfloat16)
        return torch.stack([self.cache[i] for i in ids])[None]


def represent(ids, embedding, weight, bias, variant='bf16'):
    x = embedding.get(ids)
    require(x.device.type == weight.device.type == bias.device.type == 'cpu', 'non-CPU tensor')
    if variant == 'fp32':
        return F.linear(x.float(), weight.float(), bias.float())[0,-64:].mean(0)
    if variant == 'pool_first':
        return F.linear(x[0,-64:].mean(0), weight, bias)
    require(variant == 'bf16', 'unknown numeric variant')
    return F.linear(x, weight, bias)[0,-64:].mean(0)


def q_difference(a, b):
    af, bf = a.double(), b.double()
    return dict(exact_equal=bool(torch.equal(a,b)), max_abs=float((af-bf).abs().max()),
                l2=float((af-bf).norm()), cosine=float(F.cosine_similarity(af[None],bf[None]).item()))


def scores(q, keys, ages):
    keys = keys.to(dtype=q.dtype)
    raw = F.cosine_similarity(q[None], keys, dim=-1).double().tolist()
    final = [c*math.exp(-.05*a) for c,a in zip(raw,ages)]
    ranking = order(final)
    selected = sorted([i for i in ranking if final[i] >= .05][:2])
    return dict(raw=raw, final=final, raw_order=order(raw), final_order=ranking, selected=selected,
                threshold=[s >= .05 for s in final])


def comparison(a, b):
    return dict(raw_ranking_changed=a['raw_order'] != b['raw_order'],
                final_ranking_changed=a['final_order'] != b['final_order'],
                full_top2_changed=a['selected'] != b['selected'],
                max_raw_cosine_difference=max(abs(x-y) for x,y in zip(a['raw'],b['raw'])),
                threshold_crossings=sum(x != y for x,y in zip(a['threshold'],b['threshold'])))


def clue_variant(prompt, question, donor_prompt, donor_question):
    a,b,_ = next(s for s in token_spans(prompt,question) if s[2] == 'question_clue')
    c,d,_ = next(s for s in token_spans(donor_prompt,donor_question) if s[2] == 'question_clue')
    return prompt[:a]+donor_prompt[c:d]+prompt[b:]


def candidate_variant(ids, donor_ids, categories, donor_categories):
    pos = [len(ids)-64+i for i,c in enumerate(categories) if c == 'candidate_list']
    donor_pos = [len(donor_ids)-64+i for i,c in enumerate(donor_categories) if c == 'candidate_list']
    require(len(pos) == len(donor_pos) == 17, 'candidate tail budget differs')
    changed = list(ids)
    for a,b in zip(pos,donor_pos):
        changed[a] = donor_ids[b]
    require(all(changed[i] == ids[i] for i in range(len(ids)) if i not in pos), 'noncandidate intervention')
    return changed


def geometry(q):
    require(q.ndim == 2 and len(q) >= 2 and bool(torch.isfinite(q).all()), 'bad geometry input')
    q = q.double()
    norms = q.norm(dim=1)
    unit = q/norms[:,None]
    qq = unit@unit.T
    pairs = qq[torch.triu(torch.ones_like(qq,dtype=torch.bool),diagonal=1)]
    centered = q-q.mean(0)
    variances = centered.square().mean(0)
    singular = torch.linalg.svdvals(centered)
    eigen = singular.square()/len(q)
    total = eigen.sum()
    participation = float(total.square()/eigen.square().sum()) if total > 0 else None
    return dict(n=len(q), qq_cosine=stats(pairs.tolist()), norm=stats(norms.tolist()),
                per_coordinate_population_variance=variances.tolist(), total_population_variance=float(variances.sum()),
                variance_energy_fraction=float(variances.sum()/q.square().sum(1).mean()),
                centered_covariance_participation_effective_rank=participation,
                centered_covariance_eigenvalues=eigen.tolist()), qq.float().numpy()


def numeric_summary(rows):
    variants = {}
    for name in ('clue_only','candidate_tail_only','fp32','pool_first'):
        variants[name] = dict(exact_equal=sum(r[name]['exact_equal'] for r in rows),
            q_cosine=stats([r[name]['cosine'] for r in rows]),
            q_max_abs=stats([r[name]['max_abs'] for r in rows]),
            q_l2=stats([r[name]['l2'] for r in rows]),
            full_top2_changed=sum(r[name]['full_top2_changed'] for r in rows),
            raw_ranking_changed=sum(r[name]['raw_ranking_changed'] for r in rows),
            final_ranking_changed=sum(r[name]['final_ranking_changed'] for r in rows),
            threshold_crossings=sum(r[name]['threshold_crossings'] for r in rows),
            raw_cosine_max_abs=stats([r[name]['max_raw_cosine_difference'] for r in rows]))
    return dict(groups=len(rows), original_hash_exact_matches=sum(r['original_hash_exact_match'] for r in rows),
                norm_abs_error=stats([r['norm_abs_error'] for r in rows]),
                cosine_max_abs_error=stats([r['cosine_max_abs_error'] for r in rows]),
                original_full_top2_exact_matches=sum(r['original_full_top2_exact_match'] for r in rows),
                original_raw_ranking_matches=sum(r['original_raw_ranking_match'] for r in rows),
                original_final_ranking_matches=sum(r['original_final_ranking_match'] for r in rows),
                original_threshold_crossings=sum(r['original_threshold_crossings'] for r in rows),variants=variants)


def run(output):
    start = time.monotonic()
    require(not output.exists() and output.parent == HERE, 'fresh output inside e01v required')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    parent_check = validate_parent_e01()
    before = inventory(PARENT)
    e01_before = inventory(HERE.parent/'e01')
    m = read(PARENT/'manifest.json')
    require(m['contract_sha256'] == CONTRACT, 'parent contract drift')
    def check_inputs():
        for name,h in m['source_sha256'].items():
            require(sha(ROOT/name) == h, 'source drift: '+name)
        for name,h in m['input_sha256'].items():
            require(sha(name) == h, 'input drift: '+name)
    check_inputs()
    checkpoint = Path(m['checkpoint_path'])
    environment = read(PARENT/'environment_contract.json')
    tokenizer = Tokenizer.from_file(str(Path(environment['tokenizer_path'])/'tokenizer.json'))
    other_tokenizer = Tokenizer.from_file(str(checkpoint/'tokenizer.json'))
    embedding_path = Path(next(p for p in m['input_sha256'] if p.endswith('/model.safetensors')))
    embedding = EmbeddingRows(embedding_path)
    projs = torch.load(checkpoint/'projs.bin',map_location='cpu',weights_only=True)['reasoner_to_weaver']
    weight,bias = projs['weight'],projs['bias']
    require(weight.shape == (1536,1536) and bias.shape == (1536,) and weight.dtype == bias.dtype == torch.bfloat16, 'projection contract')
    windows = {}
    with (HERE.parent/'e01/run-20261008-01/token_windows.jsonl').open() as f:
        for line in f:
            w=json.loads(line);windows[w['context_index'],w['query_id']]=w
    output.mkdir()
    atomic(output/'status.json',dict(status='RUNNING'))
    original, variants, prompts, ids, interventions = {}, {}, {}, {}, []
    tokenizer_equal = 0
    for ctx in range(5):
        for qid in range(100):
            d=read(PARENT/'jobs'/f's42-c{ctx}'/'queries'/f'q{qid:03d}_full.json')
            prompt=d['runtime_result']['rendered_query_prompt'];question=d['record']['question']
            seq=tokenizer.encode(prompt,add_special_tokens=False).ids
            require(len(seq)==d['runtime_result']['generations'][0]['input_len'], 'actual tokenizer length mismatch')
            require(digest(seq[-64:])==windows[ctx,qid]['tail_token_ids_sha256'],'E01 window mismatch')
            require(seq == other_tokenizer.encode(prompt,add_special_tokens=False).ids,'base/checkpoint tokenizer mismatch')
            tokenizer_equal += 1
            prompts[ctx,qid]=(prompt,question);ids[ctx,qid]=seq
        assignment=donors(f'E01V-c{ctx}',list(range(100)))
        for qid in range(100):
            key=ctx,qid; donor=ctx,assignment[qid]
            p,q=prompts[key];dp,dq=prompts[donor]
            changed_prompt=clue_variant(p,q,dp,dq)
            clue_ids=tokenizer.encode(changed_prompt,add_special_tokens=False).ids
            require(clue_ids[-64:] == ids[key][-64:], 'clue intervention tail changed')
            categories=[t['category'] for t in windows[key]['tokens']]
            donor_categories=[t['category'] for t in windows[donor]['tokens']]
            tail_ids=candidate_variant(ids[key],ids[donor],categories,donor_categories)
            with torch.inference_mode():
                original[key]=represent(ids[key],embedding,weight,bias)
                variants[key]={
                    'clue_only':represent(clue_ids,embedding,weight,bias),
                    'candidate_tail_only':represent(tail_ids,embedding,weight,bias),
                    'fp32':represent(ids[key],embedding,weight,bias,'fp32'),
                    'pool_first':represent(ids[key],embedding,weight,bias,'pool_first')}
            interventions.append(dict(context_index=ctx,query_id=qid,donor_query_id=donor[1],
                original_ids_sha256=digest(ids[key]),clue_ids_sha256=digest(clue_ids),candidate_ids_sha256=digest(tail_ids),
                candidate_tail_ids=tail_ids[-64:],original_tail_ids=ids[key][-64:],
                clue_prefix_changed=clue_ids[:-64]!=ids[key][:-64],
                candidate_token_changes=sum(a!=b for a,b in zip(ids[key],tail_ids))))
            require(time.monotonic()-start<1800,'30min wall cap')
        print('EXTRACTED',ctx,flush=True)
    rows=[]
    for job in m['jobs']:
        jid=job['id'];ctx=job['context_index'];directory=PARENT/'jobs'/jid
        require(sha(directory/'frozen_bank.pt') == read(directory/'snapshot_identity.json')['snapshot_file_sha256'], 'snapshot drift')
        keys,config = load_bank(directory/'frozen_bank.pt')
        require(config.pool_last_n == 64 and config.decay_alpha == .05 and config.retrieve_threshold == .05 and config.top_k == 2,'bank config drift')
        for qid in range(100):
            d=read(directory/'queries'/f'q{qid:03d}_full.json')
            saved=d['runtime_result']['query_score_decomposition'];t=d['record']['selection'][0]
            require(d['runtime_result']['rendered_query_prompt']==prompts[ctx,qid][0], 'seed prompt differs')
            q=original[ctx,qid];ages=t['ages'];oldraw=t['cosine'];oldfinal=t['final_scores']
            base=scores(q,keys,ages)
            row=dict(job_id=jid,seed=job['seed'],context_index=ctx,query_id=qid,
                     original_hash_exact_match=tensor_hash(q)==saved['query_vector_hash'],
                     norm_abs_error=abs(float(q.float().norm())-saved['query_norm']),
                     cosine_max_abs_error=max(abs(a-b) for a,b in zip(base['raw'],oldraw)),
                     original_full_top2_exact_match=base['selected']==t['selected_indices'],
                     original_raw_ranking_match=base['raw_order']==order(oldraw),
                     original_final_ranking_match=base['final_order']==order(oldfinal),
                     original_threshold_crossings=sum(a != (b>=.05) for a,b in zip(base['threshold'],oldfinal)))
            for name,v in variants[ctx,qid].items():
                row[name]={**q_difference(q,v),**comparison(base,scores(v,keys,ages))}
            rows.append(row)
        print('VERIFIED',jid,flush=True)
        require(time.monotonic()-start<1800,'30min wall cap')
    geo=[];arrays={}
    sorted_keys=sorted(original)
    arrays['original_cpu_bf16_as_fp32']=torch.stack([original[k] for k in sorted_keys]).float().numpy()
    arrays['original_cpu_bf16_bits']=torch.stack([original[k] for k in sorted_keys]).view(torch.uint16).numpy()
    for name in ('clue_only','candidate_tail_only','fp32','pool_first'):
        arrays[name]=torch.stack([variants[k][name] for k in sorted_keys]).float().numpy()
    # Retain explicit 2500-row indexing; seed duplicates are not independent representations.
    arrays['job_query_index']=np.array([[j['seed'],j['context_index'],q] for j in m['jobs'] for q in range(100)],dtype=np.int32)
    arrays['original_cpu_2500']=np.concatenate([arrays['original_cpu_bf16_as_fp32'][j['context_index']*100:(j['context_index']+1)*100] for j in m['jobs']])
    for ctx in range(5):
        g,qq=geometry(torch.stack([original[ctx,q] for q in range(100)]))
        geo.append(dict(context_index=ctx,**g));arrays[f'qq_cosine_c{ctx}']=qq
    aggregated=numeric_summary(rows)
    numerical_pass=aggregated['norm_abs_error']['max']<=.001 and aggregated['cosine_max_abs_error']['max']<=.001
    selection_pass=aggregated['original_full_top2_exact_matches']==2500
    summary=dict(status='COMPLETE' if numerical_pass and selection_pass else 'PARTIAL',
        experiment_id='M6-E01V',unique_prompts=500,groups=2500,tokenizers_identical=tokenizer_equal,
        numerical_reproduction_gate=numerical_pass,full_selection_reproduction_gate=selection_pass,
        tolerance=dict(norm_abs=.001,cosine_abs=.001),aggregate=aggregated,
        contexts=[dict(context_index=i,**numeric_summary([r for r in rows if r['context_index']==i])) for i in range(5)],
        interpretation='CPU reconstruction; exact original hash is required for bitwise identity. No semantic labels or answer effects.',
        model_forward_calls=0,gpu_used=False,annotations=0,
        next_route='Stop; report Query representation evidence and request E02 or another separately approved slice.')
    np.savez(output/'queries.npz',**arrays)
    jsonl(output/'query_metrics.jsonl',rows)
    jsonl(output/'interventions.jsonl',interventions)
    atomic(output/'geometry.json',geo)
    atomic(output/'summary.json',summary)
    require(before==inventory(PARENT) and e01_before==inventory(HERE.parent/'e01'),'parent/E01 inventory drift')
    check_inputs()
    validate_parent_e01()
    require('transformers' not in sys.modules and not torch.cuda.is_initialized(),'model import/CUDA initialized')
    core={n:sha(output/n) for n in (*CORE,'interventions.jsonl')}
    elapsed=time.monotonic()-start;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    require(elapsed<1800 and rss<=16*1024**2,'execution budget exceeded')
    atomic(output/'manifest.json',dict(schema='M6-E01V/v1',parent_contract=CONTRACT,
        parent_manifest_sha256=sha(PARENT/'manifest.json'),parent_e01_acceptance_sha256=sha(HERE.parent/'e01/acceptance.json'),
        execution_contract_sha256=sha(HERE.parent/'E01V_EXECUTION.md'),analyzer_sha256=sha(__file__),
        input_sha256=m['input_sha256'],source_sha256=m['source_sha256'],core_artifact_sha256=core,
        parent_inventory_unchanged=True,e01_inventory_unchanged=True,embedding_tensor=embedding.metadata,
        embedding_rows_copied=len(embedding.cache),transformer_tensors_loaded=0,
        torch_version=torch.__version__,python_executable=sys.executable,threads=torch.get_num_threads(),
        elapsed_seconds=elapsed,max_rss_kib=rss,cuda_initialized=False,model_forward_calls=0,
        selected_outline_ref=None,paper_role='appendix',section_id='retrieval',item_id='E01V',claim_links=['C02','R04']))
    size=sum(p.stat().st_size for p in output.iterdir())
    require(size<=100*1024**2,'output cap exceeded')
    atomic(output/'status.json',dict(status=summary['status'],output_bytes=size))
    print(json.dumps(dict(status=summary['status'],numerical_pass=numerical_pass,selection_pass=selection_pass,
                         elapsed_seconds=elapsed,max_rss_kib=rss,core_hashes=core)),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve();existed=output.exists()
    try:
        run(output)
    except Exception as exc:
        if not existed and output.is_dir():
            atomic(output/'status.json',dict(status='FAILED',error=f'{type(exc).__name__}: {exc}'))
        raise
