"""M6-E01: model-free CPU diagnostics of frozen M3 records and tokenizer offsets."""
import argparse
import collections
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import statistics
import sys
import tempfile
import time
import tokenize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
M4 = HERE.parents[1] / 'm4'
CONTRACT = 'a548306d4131e4df0cba1340d3272a4882feae1d9c993c881e281f4551e433c9'
POLICIES = ('full', 'random', 'last_written', 'cosine_only', 'recency_only', 'native_order')
CATEGORIES = ('question_clue', 'candidate_header', 'candidate_list', 'answer_instruction',
              'chat_tail', 'prefix_template', 'boundary_mixed', 'unmapped')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def jsonl(path, rows):
    with Path(path).open('x') as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False)+'\n')


def inventory(root):
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(root.rglob('*')) if p.is_file()}


def stats(values):
    v = sorted(values)
    if not v:
        return dict(n=0, min=None, max=None, mean=None, population_std=None, quantiles=None)
    require(all(math.isfinite(x) for x in v), 'nonfinite measurement')
    def quantile(p):
        at = (len(v)-1)*p
        lo, hi = math.floor(at), math.ceil(at)
        return v[lo] + (v[hi]-v[lo])*(at-lo)
    return dict(n=len(v), min=v[0], max=v[-1], mean=statistics.mean(v),
                population_std=statistics.pstdev(v),
                quantiles={str(p): quantile(p) for p in (0, .05, .25, .5, .75, .95, 1)})


def order(values):
    require(all(math.isfinite(x) for x in values), 'nonfinite ranking')
    return sorted(range(len(values)), key=lambda i: (-values[i], i))


def midranks(values):
    result = [None]*len(values)
    ranked = sorted(range(len(values)), key=lambda i: values[i])
    start = 0
    while start < len(ranked):
        end = start+1
        while end < len(ranked) and values[ranked[end]] == values[ranked[start]]:
            end += 1
        rank = (start+end-1)/2+1
        for i in ranked[start:end]:
            result[i] = rank
        start = end
    return result


def correlations(x, y):
    require(len(x) == len(y) and len(x) >= 2, 'incomplete rank pair')
    require(all(math.isfinite(v) for v in x+y), 'nonfinite rank pair')
    rx, ry = midranks(x), midranks(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    xx, yy = sum((v-mx)**2 for v in rx), sum((v-my)**2 for v in ry)
    spearman = sum((a-mx)*(b-my) for a,b in zip(rx,ry))/math.sqrt(xx*yy) if xx*yy else None
    c = d = tx = ty = both = 0
    for i,j in itertools.combinations(range(len(x)),2):
        dx, dy = x[i]-x[j], y[i]-y[j]
        if dx == 0 and dy == 0:
            both += 1
        elif dx == 0:
            tx += 1
        elif dy == 0:
            ty += 1
        elif dx*dy > 0:
            c += 1
        else:
            d += 1
    denominator = math.sqrt((c+d+tx)*(c+d+ty))
    return dict(spearman=spearman, kendall_tau_b=(c-d)/denominator if denominator else None,
                concordant=c, discordant=d, ties_x_only=tx, ties_y_only=ty, ties_both=both)


def reconstruct(cosine, ages, alpha=.05, threshold=.05, top_k=2):
    require(len(cosine) == len(ages) and len(cosine) >= 3, 'incomplete bank arrays')
    require(all(a >= 0 for a in ages), 'negative age')
    scores = [c*math.exp(-alpha*a) for c,a in zip(cosine,ages)]
    native = [i for i in order(scores) if scores[i] >= threshold][:top_k]
    return scores, native, sorted(native)


def donors(job_id, query_ids):
    require(len(query_ids) == len(set(query_ids)) and len(query_ids) >= 2, 'invalid donor identities')
    ids = sorted(query_ids, key=lambda q: hashlib.sha256(f'M6-E01-v1:{job_id}:{q}'.encode()).hexdigest())
    return {q: ids[(i+1) % len(ids)] for i,q in enumerate(ids)}


def token_spans(prompt, question):
    marker = 'Below is a list of possible subsequent events:'
    qstart = prompt.find(question)
    require(qstart >= 0 and prompt.count(question) == 1, 'rendered question missing/ambiguous')
    header = prompt.find(marker, qstart)
    require(header >= qstart, 'candidate marker missing')
    start = prompt.find('[', header+len(marker))
    require(start >= 0, 'candidate list missing')
    tail = prompt[start:]
    depth, end = 0, None
    for token in tokenize.generate_tokens(io.StringIO(tail).readline):
        if token.type != tokenize.OP:
            continue
        if token.string == '[':
            depth += 1
        elif token.string == ']':
            depth -= 1
            if depth == 0:
                line,col = token.end
                end = start + sum(len(s) for s in tail.splitlines(keepends=True)[:line-1]) + col
                break
    require(end is not None, 'unterminated candidate literal')
    chat = prompt.find('<|im_end|>', end)
    chat = len(prompt) if chat < 0 else chat
    return [(0,qstart,'prefix_template'), (qstart,header,'question_clue'),
            (header,start,'candidate_header'), (start,end,'candidate_list'),
            (end,chat,'answer_instruction'), (chat,len(prompt),'chat_tail')]


def window(tokenizer, prompt, question, recorded_length, pool=64):
    enc = tokenizer.encode(prompt, add_special_tokens=False)
    require(len(enc.ids) == recorded_length, 'tokenized length differs: clipping/padding cannot be assumed')
    spans = token_spans(prompt,question)
    counts = collections.Counter({k:0 for k in CATEGORIES})
    tokens = []
    for index in range(max(0,len(enc.ids)-pool),len(enc.ids)):
        a,b = enc.offsets[index]
        labels = [name for lo,hi,name in spans if max(a,lo) < min(b,hi)]
        label = labels[0] if len(labels)==1 else 'boundary_mixed' if labels else 'unmapped'
        counts[label] += 1
        tokens.append(dict(id=enc.ids[index], offset=[a,b], category=label, text=prompt[a:b]))
    require(sum(counts.values()) == min(pool,len(enc.ids)), 'window categories not exhaustive')
    return dict(prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), input_tokens=len(enc.ids),
                window_tokens=len(tokens), counts=dict(counts), tokens=tokens,
                tail_token_ids_sha256=digest(enc.ids[-pool:]), tokenization_length_matches=True,
                pad_token_occurrences_in_rendered_encoding=sum(
                    i == tokenizer.token_to_id('<|endoftext|>') for i in enc.ids),
                length_based_clipping_detected=False,
                limitation='Saved input token IDs unavailable; matching length does not establish ID equality.')


def summarize(rows):
    metric_keys = ('query_norm','cosine_min','cosine_max','cosine_range','cosine_top12_margin',
                   'cosine_top23_margin','final_top12_margin','final_top23_margin',
                   'raw_final_spearman','raw_final_tau_b','donor_raw_spearman','donor_raw_tau_b',
                   'donor_final_spearman','donor_final_tau_b','raw_final_discordant','raw_final_ties_both')
    bool_keys = ('raw_ranking_changed_by_donor','final_ranking_changed_by_donor',
                 'semantic_top2_changed_by_donor','full_top2_changed_by_donor','raw_final_top2_match')
    return dict(groups=len(rows), distributions={k:stats([r[k] for r in rows if r[k] is not None])
                 for k in metric_keys}, counts={k:sum(r[k] for r in rows) for k in bool_keys},
                score_distributions={k:stats([v for r in rows for v in r[k]])
                                     for k in ('raw_cosines','decayed_scores')},
                token_counts={k:sum(r['window_counts'][k] for r in rows) for k in CATEGORIES})


def run(parent, output):
    start = time.monotonic()
    require(not output.exists() and output.parent == HERE, 'fresh output directly inside e01 required')
    before = inventory(parent)
    old_before = inventory(M4)
    manifest = read(parent/'manifest.json')
    require(manifest['contract_sha256'] == CONTRACT, 'wrong parent contract')
    require(read(parent/'results.json')['status'] == 'COMPLETE', 'incomplete parent results')
    require(collections.Counter(j['status'] for j in read(parent/'queue.json')['jobs'].values()) == {'done':25}, 'incomplete queue')
    for name,h in manifest['source_sha256'].items():
        require(sha(ROOT/name) == h, 'frozen source drift: '+name)
    frozen = read(M4/'run-20261008-a1/manifest.json')['parent_json_sha256']
    for name,h in frozen.items():
        require(sha(parent/name) == h, 'parent JSON drift: '+name)
    for receipt in ['acceptance.json','acceptance_b1.json','acceptance_b2.json']:
        r = read(M4/receipt)
        require(r['status']=='PASS', 'M4 not accepted')
        for path in r.get('runs',r.get('compared_runs')):
            for name,h in r['identical_artifact_sha256'].items():
                require(sha(ROOT/path/name) == h, 'M4 artifact drift: '+name)
    # This import is tokenizer-only. Never import torch, transformers, or a model loader.
    from tokenizers import Tokenizer
    tokenizer_path = parent_tokenizer = Path(manifest['checkpoint_path'])/'tokenizer.json'
    require(str(tokenizer_path) in manifest['input_sha256'], 'tokenizer not in frozen inputs')
    require(sha(tokenizer_path)==manifest['input_sha256'][str(tokenizer_path)], 'tokenizer drift')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    b1 = {}
    with (M4/'run-20261008-b1/diagnostics.jsonl').open() as f:
        for line in f:
            d=json.loads(line); key=(d['job_id'],d['query_id'])
            require(key not in b1,'duplicate B1 row');b1[key]=d
    require(len(b1)==2500, 'incomplete B1 index')
    output.mkdir()
    atomic(output/'status.json',dict(status='RUNNING'))
    rows, jobs, windows, signatures = [], [], {}, {}
    for job in manifest['jobs']:
        jid = job['id']; directory=parent/'jobs'/jid
        ident=read(directory/'snapshot_identity.json')
        require(sha(directory/'frozen_bank.pt')==ident['snapshot_file_sha256'],'bank hash drift')
        require(read(directory/'context_identity.json')['bank_config']['pool_last_n']==64,'pool drift')
        bank = {}
        for q in range(100):
            traces=[]; envelope=None
            for policy in POLICIES:
                d=read(directory/'queries'/f'q{q:03d}_{policy}.json')
                signed=dict(d); checksum=signed.pop('content_sha256')
                require(digest(signed)==checksum,'raw checksum mismatch')
                require(d['identity']==dict(campaign=manifest['campaign'],seed=job['seed'],
                        context_index=job['context_index'],query_id=q,policy=policy),'raw identity mismatch')
                require(d['complete'] and d['provenance']==ident,'incomplete/wrong snapshot record')
                record=d['record'];runtime=d['runtime_result']
                require(record['query_write_count_delta']==0 and record['snapshot_unchanged'],'query wrote bank')
                require(len(record['selection'])==1,'multiple retrieval turns')
                traces.append(record['selection'][0])
                signature=digest([record['question'],record['gold_answers'],record['query_sha256'],runtime['rendered_query_prompt']])
                key=(job['context_index'],q)
                require(key not in signatures or signatures[key]==signature,'question/prompt drift across policies/seeds')
                signatures[key]=signature
                if policy=='full':envelope=d
            base=traces[0]
            for t in traces:
                for name in ('cosine','ages','last_write','final_scores','decay','native_indices','n'):
                    require(t[name]==base[name],'unpaired selector bases')
                require(t['n']==2 and t['latent_count']==16,'unexpected selection budget')
            record=envelope['record'];runtime=envelope['runtime_result'];decomp=runtime['query_score_decomposition']
            old=b1[jid,q]
            require(base['cosine']==old['cosine'] and base['ages']==old['access_ages'],'B1 base drift')
            scores,native,full=reconstruct(base['cosine'],base['ages'])
            require(scores==base['final_scores'] and native==old['native_order'] and full==old['full'],'score/selector mismatch')
            require([s['raw_cosine'] for s in decomp['slots']]==base['cosine'],'native/adapter cosine mismatch')
            require(decomp['query_cosine_to_first'] is None and decomp['query_cosine_to_previous'] is None,'changed vector schema')
            selections={t['policy']:t['selected_indices'] for t in traces}
            expected=dict(full=old['full'],cosine_only=old['cosine_only'],recency_only=old['recency'],last_written=old['last_written'],native_order=old['native_order'])
            expected['random'] = sorted(random.Random(traces[1]['random_seed']).sample(range(len(scores)), 2))
            for name,selected in selections.items():
                require(len(selected)==len(set(selected))==2 and all(0<=i<len(scores) for i in selected),'invalid selection')
                if name in expected:require(selected==expected[name],'B1 selected indices drift')
            require(len(runtime['generations'])==1,'multiple generations')
            wkey=(job['context_index'],q);prompt=runtime['rendered_query_prompt'];length=runtime['generations'][0]['input_len']
            if wkey not in windows:
                windows[wkey]=dict(context_index=wkey[0],query_id=q,**window(tokenizer,prompt,record['question'],length))
            else:require(windows[wkey]['input_tokens']==length,'input length drift across seed')
            bank[q]=dict(cosine=base['cosine'],ages=base['ages'],scores=scores,selections=selections,
                         norm=decomp['query_norm'],hash=decomp['query_vector_hash'],window=windows[wkey])
        require(len({tuple(v['ages']) for v in bank.values()})==1,'bank age changes across queries')
        assignment=donors(jid,list(bank));local=[]
        for q,v in bank.items():
            other=bank[assignment[q]]; donor_scores,_,donor_full=reconstruct(other['cosine'],v['ages'])
            raw,final=v['cosine'],v['scores'];raw_order,final_order=order(raw),order(final)
            rf=correlations(raw,final);dr=correlations(raw,other['cosine']);df=correlations(final,donor_scores)
            row=dict(job_id=jid,seed=job['seed'],context_index=job['context_index'],query_id=q,
                     donor_query_id=assignment[q],query_norm=v['norm'],raw_cosines=raw,decayed_scores=final,
                     cosine_min=min(raw),cosine_max=max(raw),
                     cosine_range=max(raw)-min(raw),cosine_top12_margin=raw[raw_order[0]]-raw[raw_order[1]],
                     cosine_top23_margin=raw[raw_order[1]]-raw[raw_order[2]],
                     final_top12_margin=final[final_order[0]]-final[final_order[1]],
                     final_top23_margin=final[final_order[1]]-final[final_order[2]],
                     raw_final_spearman=rf['spearman'],raw_final_tau_b=rf['kendall_tau_b'],
                     raw_final_discordant=rf['discordant'],raw_final_ties_both=rf['ties_both'],
                     donor_raw_spearman=dr['spearman'],donor_raw_tau_b=dr['kendall_tau_b'],
                     donor_final_spearman=df['spearman'],donor_final_tau_b=df['kendall_tau_b'],
                     raw_ranking_changed_by_donor=raw_order!=order(other['cosine']),
                     final_ranking_changed_by_donor=final_order!=order(donor_scores),
                     semantic_top2_changed_by_donor=sorted(raw_order[:2])!=sorted(order(other['cosine'])[:2]),
                     full_top2_changed_by_donor=v['selections']['full']!=donor_full,
                     raw_final_top2_match=sorted(raw_order[:2])==v['selections']['full'],
                     window_counts=v['window']['counts'])
            rows.append(row);local.append(row)
        controls={}
        for policy in POLICIES:
            seq=[v['selections'][policy] for v in bank.values()]
            intersections=[len(set(s)&set(v['selections']['full'])) for s,v in zip(seq,bank.values())]
            controls[policy]=dict(slot_frequency=dict(sorted(collections.Counter(i for s in seq for i in s).items())),
                                 set_combinations=dict(sorted(collections.Counter(','.join(map(str,sorted(s))) for s in seq).items())),
                                 ordered_combinations=dict(sorted(collections.Counter(','.join(map(str,s)) for s in seq).items())),
                                 full_exact_set_matches=sum(n==2 for n in intersections),
                                 full_intersection_slots=sum(intersections),full_jaccard=stats([n/(4-n) for n in intersections]))
        jobs.append(dict(job_id=jid,seed=job['seed'],context_index=job['context_index'],
                         unique_query_hashes=len({v['hash'] for v in bank.values()}),
                         unique_raw_rankings=len({tuple(order(v['cosine'])) for v in bank.values()}),
                         unique_final_rankings=len({tuple(order(v['scores'])) for v in bank.values()}),
                         unique_raw_top2_sets=len({tuple(sorted(order(v['cosine'])[:2])) for v in bank.values()}),
                         controls=controls,**summarize(local)))
        print('VERIFIED',jid,flush=True)
        require(time.monotonic()-start<1800,'30min resource cap exceeded')
    # Source names only; no human evidence annotations or semantic matching.
    import pyarrow.parquet as pq
    data_path=Path(next(p for p in manifest['input_sha256'] if p.endswith('.parquet')))
    require(sha(data_path)==manifest['input_sha256'][str(data_path)],'dataset identity drift')
    sources=[]
    for meta in pq.read_table(data_path,columns=['metadata'])['metadata'].to_pylist():
        ids=meta.get('qa_pair_ids') or []
        if ids and str(ids[0]).startswith('eventqa_65536_'):
            sources.append(dict(source=meta.get('source'),qa_pair_ids_sha256=digest(ids)))
    require(len(sources)==5,'source context inventory mismatch')
    summary=dict(status='COMPLETE_WITH_SEMANTIC_LIMITATIONS',groups=len(rows),unique_questions=len(windows),
                 jobs=len(jobs),policy_records_verified=len(rows)*6,aggregate=summarize(rows),
                 contexts=[dict(context_index=i,**summarize([r for r in rows if r['context_index']==i])) for i in range(5)],
                 source_metadata=sources,source_names_unique=len({s['source'] for s in sources}),
                 blocked=dict(semantic_permutation='No verified evidence spans; donor is mechanical, not a semantic negative.',
                              semantic_retrieval_metrics='Unknown relevance labels; no semantic Recall/MRR.',
                              query_geometry='Complete q vectors missing; no QQ cosine or per-coordinate variance.'),
                 new_model_runs=0,annotations=0,full_question_answer_metrics='Reused; no new answers or rescoring')
    jsonl(output/'query_metrics.jsonl',rows)
    jsonl(output/'token_windows.jsonl',[windows[k] for k in sorted(windows)])
    atomic(output/'by_job.json',jobs)
    atomic(output/'summary.json',summary)
    require(before==inventory(parent) and old_before==inventory(M4),'parent inventory changed')
    for name,h in frozen.items():require(sha(parent/name)==h,'parent JSON changed during run')
    require('torch' not in sys.modules and 'transformers' not in sys.modules,'model dependency imported')
    core={name:sha(output/name) for name in ('query_metrics.jsonl','token_windows.jsonl','by_job.json','summary.json')}
    atomic(output/'manifest.json',dict(schema='M6-E01/v1',parent_contract_sha256=CONTRACT,
         parent_manifest_sha256=sha(parent/'manifest.json'),parent_json_hash_count=len(frozen),
         parent_json_sha256=frozen,parent_inventory_unchanged=True,m4_inventory_unchanged=True,
         analyzer_sha256=sha(__file__),execution_contract_sha256=sha(HERE.parent/'E01_EXECUTION.md'),
         tokenizer_sha256=sha(parent_tokenizer),tokenizer_path=str(parent_tokenizer),
         core_artifact_sha256=core,python_executable=sys.executable,python_version=platform.python_version(),
         elapsed_seconds=time.monotonic()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         torch_imported=False,transformers_imported=False,gpu_used=False,model_runs=0,
         exp_id='M6-E01',item_id='E01',section_id='retrieval',paper_role='main limitation/diagnostics',
         claim_links=['C02','R03','R04','R07'],next_route='Request E02 annotation pilot or E01V; no automatic execution'))
    total=sum(p.stat().st_size for p in output.iterdir())
    require(total<=100*1024**2,'output budget exceeded')
    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<=4*1024**2,'RAM budget exceeded')
    atomic(output/'status.json',dict(status='COMPLETE',output_bytes=total))
    print(json.dumps(dict(status='COMPLETE',groups=len(rows),output_bytes=total,core_hashes=core)),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',type=Path,default=ROOT/'outputs/mab/review_m3_formal/20261002-m3-six-policy-v3')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output_existed = args.output.exists()
    try:
        run(args.parent.resolve(),args.output.resolve())
    except Exception as exc:
        if not output_existed and args.output.is_dir():
            atomic(args.output/'status.json',dict(status='FAILED',error=f'{type(exc).__name__}: {exc}'))
        raise
