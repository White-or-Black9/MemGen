"""CPU/std-library offline failure audit; no new predictions or scorer changes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import run_content as p
from scripts.eval import review_eventqa_decomposition as metrics

HERE=Path(__file__).resolve().parent
RUN=HERE/'pilot-20261008-01'
PROVENANCE=p.ROOT/'research_notes/next_version/m4/run-20261008-a1/provenance.jsonl'


def diagnostic(prediction,candidates,official):
    """Posthoc diagnostics only: not an alternative scoring/parser contract."""
    lines=[line.strip() for line in prediction.splitlines() if line.strip()]
    normalized=official.normalize_answer(prediction)
    mentions=[i for i,answer in enumerate(candidates) if official.normalize_answer(answer) in normalized]
    return dict(last_nonempty_line=lines[-1] if lines else '',leading_nonempty_lines=lines[:-1],
        raw_candidate_mentions=mentions,unique_candidate_mention=mentions[0] if len(mentions)==1 else None,
        diagnostic_only=True)


def compare_bank_snapshots(a,b):
    if a['combined_frozen_bank_hash']!=b['combined_frozen_bank_hash']:
        raise ValueError('Combined frozen bank differs')
    sa={s['slot_index']:s for s in a['slots']};sb={s['slot_index']:s for s in b['slots']}
    if set(sa)!=set(sb):raise ValueError('Snapshot slots differ')
    for i in sa:
        for key in ['memory_tensor_hash','key_tensor_hash','memory_shape','key_shape','memory_dtype','key_dtype']:
            if sa[i][key]!=sb[i][key]:raise ValueError(f'Snapshot tensor differs: {i}/{key}')
    return True


def run(output):
    if output.exists():raise ValueError('Output exists; preserve prior audit')
    started=time.monotonic()
    files=[x for base in [RUN,p.E03] for x in base.rglob('*') if x.is_file()]
    files += [PROVENANCE,p.E02/'source_slot_mapping.jsonl',p.HERE.parent/'E04_EXECUTION.md',
              Path(__file__),HERE/'test_analyze_failures.py',metrics.OFFICIAL]
    before={str(f):p.c.sha(f) for f in files}
    p.verify_inputs(p.c.read(RUN/'manifest.json'))
    manifest=p.c.read(RUN/'manifest.json');result=p.c.read(RUN/'results.json')
    if result['status']!='COMPLETE_EXPLORATORY_CONTENT_PILOT':raise ValueError('E04 incomplete')
    official=metrics.load_official()
    groups={arm:{r['sample_id']:r for r in result['records'] if r['arm']==arm}
            for arm in ['original','source','random']}
    provenance={(r['seed'],r['context_index'],r['slot_index']):r for r in p.e3.jsonl(PROVENANCE)}
    mappings={r['sample_id']:r for r in p.e3.jsonl(p.E02/'source_slot_mapping.jsonl') if r['seed']==42}
    details=[]
    for plan in manifest['plans']:
        sid=plan['sample_id'];ctx,q=plan['context_index'],plan['query_id']
        a,b=groups['original'][sid],groups['source'][sid]
        candidates=metrics.candidates(a['question'])
        ad=diagnostic(a['prediction'],candidates,official);bd=diagnostic(b['prediction'],candidates,official)
        original=p.c.read(RUN/'queries'/f'{sid}_original_replay_r0.json')
        source=p.c.read(RUN/'queries'/f'{sid}_source_r0.json')
        repeat=p.c.read(RUN/'queries'/f'{sid}_source_r1.json')
        parent=p.e3.checked_envelope(p.PARENT/f'jobs/s42-c{ctx}/queries/q{q:03d}_full.json')
        snapshots_match=all(compare_bank_snapshots(parent['runtime_result']['pre_query_bank_tensor_snapshot'],
            e['runtime_result']['pre_query_bank_tensor_snapshot']) for e in [original,source,repeat])
        ot=original['record']['tensor_trace'][0]['tensor_details']
        bt=source['record']['tensor_trace'][0]['tensor_details']
        if ot[0]['memory_sha256']!=bt[0]['memory_sha256'] or ot[1]['memory_sha256']==bt[1]['memory_sha256']:
            raise ValueError('Fixed filler / changed second tensor contract failed')
        audits=[]
        for label,envelope,indices in [('original',original,plan['original_indices']),('source',source,plan['source_indices'])]:
            rt=envelope['runtime_result'];record=envelope['record'];g=rt['generations'][-1]
            meta=record['tensor_trace'][0]['tensor_details']
            for position,(index,tensor) in enumerate(zip(indices,meta)):
                prov=provenance[(42,ctx,index)]
                mapped=next(s for s in mappings[sid]['slots'] if s['slot_index']==index)
                if prov['slot_version']!=mapped['slot_version'] or prov['direct_source_chunk']!=mapped['direct_source_chunk']:
                    raise ValueError('M4 provenance / E02 slot version differs')
                if label=='source' and position==1 and prov['slot_version']!=plan['source_slot_version']:
                    raise ValueError('Selected source version differs from frozen plan')
                # Age must use native pre-retrieval trace, not post-access returned-slot metadata.
                audits.append(dict(arm=label,position=position,slot_index=index,slot_version=prov['slot_version'],
                    direct_source_chunk=prov['direct_source_chunk'],indirect_possible=prov['indirect_possible_source_chunks'],
                    source_link=mapped['source_link'],memory_shape=tensor['memory_shape'],
                    memory_sha256=tensor['memory_sha256'],key_sha256=tensor['key_sha256'],
                    expected_hash_match=tensor['memory_sha256']==tensor['expected_memory_sha256'],
                    memory_norm=tensor['memory_norm'],key_norm=tensor['key_norm'],
                    pre_retrieval_age=record['native_full_selection'][0]['ages'][index],
                    score=record['native_full_selection'][0]['final_scores'][index],
                    output_token_count=g['output_len'],at_generation_cap=g['output_len']>=40))
        tail_equal=ad['last_nonempty_line']==bd['last_nonempty_line']
        changed=a['prediction']!=b['prediction']
        details.append(dict(sample_id=sid,context_index=ctx,query_id=q,
            gold_answers=a['gold_answers'],candidate_list=candidates,
            original={k:a[k] for k in ['prediction','parsed','candidate_valid','strict_candidate_correct','official_raw_recall','raw_gold_any']},
            source={k:b[k] for k in ['prediction','parsed','candidate_valid','strict_candidate_correct','official_raw_recall','raw_gold_any']},
            original_diagnostic=ad,source_diagnostic=bd,text_changed=changed,
            exact_last_line_equal=tail_equal,raw_unique_candidate_same=(ad['unique_candidate_mention'] is not None and
                ad['unique_candidate_mention']==bd['unique_candidate_mention']),
            observable_bucket='prefix_only_change_same_last_line' if changed and tail_equal else
                'exact_output_unchanged' if not changed else 'other_change',
            strict_regression=bool(a['strict_candidate_correct'] and not b['strict_candidate_correct']),
            parent_snapshot_tensor_hashes_match=snapshots_match,tensor_and_provenance_audit=audits,
            source_repeat_prediction_equal=source['record']['prediction']==repeat['record']['prediction']))
    e03=p.c.read(p.E03/'results.json');e03_rows=[r for r in e03['records'] if r['repeat']==0]
    e03_details=[]
    for sid in sorted({r['sample_id'] for r in e03_rows}):
        on=next(r for r in e03_rows if r['sample_id']==sid and r['arm']=='on')
        off=next(r for r in e03_rows if r['sample_id']==sid and r['arm']=='off')
        candidates=metrics.candidates(on['question']);od=diagnostic(on['prediction'],candidates,official)
        fd=diagnostic(off['prediction'],candidates,official)
        e03_details.append(dict(sample_id=sid,on_prediction=on['prediction'],off_prediction=off['prediction'],
            on_diagnostic=od,off_diagnostic=fd,on_strict=on['strict_candidate_correct'],off_strict=off['strict_candidate_correct'],
            on_raw_recall=on['official_raw_recall'],off_raw_recall=off['official_raw_recall'],
            raw_unique_candidate_same=(od['unique_candidate_mention'] is not None and
                od['unique_candidate_mention']==fd['unique_candidate_mention'])))
    summary=dict(status='OFFLINE_FAILURE_AUDIT_COMPLETE',all_e04_questions=6,original_pilot_denominator=10,
        buckets=dict(Counter(r['observable_bucket'] for r in details)),
        exact_last_line_equal_count=sum(r['exact_last_line_equal'] for r in details),
        raw_unique_candidate_same_count=sum(r['raw_unique_candidate_same'] for r in details),
        strict_regression_ids=[r['sample_id'] for r in details if r['strict_regression']],
        candidate_valid_original=sum(r['original']['candidate_valid'] for r in details),
        candidate_valid_source=sum(r['source']['candidate_valid'] for r in details),
        raw_gold_any_original=sum(r['original']['raw_gold_any'] for r in details),
        raw_gold_any_source=sum(r['source']['raw_gold_any'] for r in details),
        e03_raw_unique_candidate_same_count=sum(r['raw_unique_candidate_same'] for r in e03_details),
        e03_raw_recall_on=e03['summary']['on']['raw_recall'],e03_raw_recall_off=e03['summary']['off']['raw_recall'],
        e03_improved_ids=[r['sample_id'] for r in e03_details if r['on_strict'] and not r['off_strict']],
        parser_sha256=p.c.sha(metrics.OFFICIAL),new_gpu_runs=0,semantic_retention='UNKNOWN',
        next_route='E05 planning only after approval; no automatic GPU launch')
    after={str(f):p.c.sha(f) for f in files}
    if before!=after:raise ValueError('Parent evidence changed during audit')
    output.mkdir(parents=True)
    p.c.atomic(output/'e04_cases.json',details);p.c.atomic(output/'e03_cases.json',e03_details)
    p.c.atomic(output/'summary.json',summary)
    p.c.atomic(output/'acceptance.json',dict(status='PASS_OFFLINE_AUDIT',inputs_preserved=True,
        input_sha256=before,elapsed_seconds=time.monotonic()-started,new_gpu_runs=0,
        core_sha256={n:p.c.sha(output/n) for n in ['e04_cases.json','e03_cases.json','summary.json']}))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output.resolve())
