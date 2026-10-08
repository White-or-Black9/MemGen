"""Read-only full artifact validation; no GPU/model imports."""
import argparse
import json
from pathlib import Path
import run_content as p


def validate(output):
    manifest=p.c.read(output/'manifest.json');p.verify_inputs(manifest)
    status=p.c.read(output/'status.json')
    if status['status']!='COMPLETE_EXPLORATORY_CONTENT_PILOT':
        raise ValueError(f'Not complete: {status["status"]}')
    files=list((output/'queries').glob('*.json'))
    if len(files)!=18 or status['completed_generations']!=18:
        raise ValueError('Generation count differs')
    for plan in manifest['plans']:
        sid=plan['sample_id'];ctx,q=plan['context_index'],plan['query_id']
        old=p.e3.checked_envelope(p.PARENT/f'jobs/s42-c{ctx}/queries/q{q:03d}_full.json')
        read={}
        for arm,repeat in [('original_replay',0),('source',0),('source',1)]:
            e=p.c.read(output/'queries'/f'{sid}_{arm}_r{repeat}.json')
            unsigned=dict(e);digest=unsigned.pop('content_sha256')
            if p.c.digest(unsigned)!=digest:raise ValueError('Envelope hash differs')
            if e['contract_sha256']!=manifest['contract_sha256']:raise ValueError('Contract differs')
            r=e['record'];result=e['runtime_result'];trace=r['native_full_selection']
            if (r['sample_id'],r['context_index'],r['query_id'],r['seed'],r['arm'],r['repeat'])!=(sid,ctx,q,42,arm,repeat):
                raise ValueError('Answer identity differs')
            expected=plan['original_indices'] if arm=='original_replay' else plan['source_indices']
            if r['actual_indices']!=expected or result['generations'][-1]['retrieved_indices']!=expected:
                raise ValueError('Wrong memory ordering')
            p.e3.validate_generation(result,trace,'on')
            if trace!=old['record']['selection']:raise ValueError('Native retrieval drift')
            if result['rendered_query_prompt']!=old['runtime_result']['rendered_query_prompt']:
                raise ValueError('Prompt changed')
            if not r['snapshot_unchanged'] or result['bank_snapshot_changed_after_query'] or result['query_write_count_delta']!=0:
                raise ValueError('Bank changed')
            tensor_trace=r['tensor_trace']
            if len(tensor_trace)!=1 or tensor_trace[0]['actual_indices']!=expected:
                raise ValueError('Wrong tensor trace')
            for detail,i in zip(tensor_trace[0]['tensor_details'],expected):
                if detail['slot_index']!=i or detail['memory_shape']!=[8,1536]:
                    raise ValueError('Tensor shape/slot mismatch')
                if detail['memory_sha256']!=detail['expected_memory_sha256']:
                    raise ValueError('Tensor hash mismatch')
            if arm=='original_replay' and r['prediction']!=old['record']['prediction']:
                raise ValueError('Reuse bridge failed')
            read[(arm,repeat)]=r
        a,b=read[('source',0)],read[('source',1)]
        if a['prediction']!=b['prediction'] or a['tensor_trace']!=b['tensor_trace']:
            raise ValueError('Source repeat unstable')
    from scripts.eval import review_eventqa_decomposition as m
    result=p.c.read(output/'results.json');official=m.load_official()
    groups={a:{} for a in ['original','source','random']}
    for r in result['records']:
        expected=m.evaluate(r['prediction'],r['gold_answers'],m.candidate_info(r['question'],r['gold_answers'],official),r['format_flags'],official)
        if any(r[k]!=v for k,v in expected.items()):raise ValueError('Metric replay differs')
        sid=r['sample_id'];arm=r['arm']
        if sid in groups[arm]:raise ValueError('Duplicate primary answer')
        groups[arm][sid]=r
        plan=next(x for x in manifest['plans'] if x['sample_id']==sid)
        if arm=='source':expected_record=p.c.read(output/'queries'/f'{sid}_source_r0.json')['record']
        else:
            policy='full' if arm=='original' else 'random'
            expected_record=p.e3.checked_envelope(p.PARENT/f"jobs/s42-c{plan['context_index']}/queries/q{plan['query_id']:03d}_{policy}.json")['record']
        if any(r[k]!=expected_record[k] for k in ['prediction','question','gold_answers']):
            raise ValueError('Primary answer differs from specified source')
    if any(len(g)!=6 for g in groups.values()):raise ValueError('Primary coverage differs')
    for arm,g in groups.items():
        summary=m.summarize([dict(r,official_em=r['official_em'] or 0,official_raw_recall=r['official_raw_recall'] or 0) for r in g.values()])
        if summary!=result['summary'][arm]:raise ValueError('Summary differs')
    if p.paired(groups['source'],groups['original'])!=result['paired_source_original']:
        raise ValueError('Primary paired table differs')
    if p.paired(groups['source'],groups['random'])!=result['paired_source_random_auxiliary']:
        raise ValueError('Auxiliary paired table differs')
    if result['formal_E02_pass'] or result['semantic_negative'] or result['efficacy_gate']!='UNDETERMINED':
        raise ValueError('Unsupported semantic/gate claim')
    print(json.dumps(dict(status='PASS_TECHNICAL_EXPLORATORY_ONLY',new_envelopes=18,
        primary_answers=18,original_bridge_matches=6,source_repeat_matches=6,
        all_metrics_replayed=True,all_inputs_preserved=True,semantic_gate_pass=False),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    validate(parser.parse_args().output.resolve())
