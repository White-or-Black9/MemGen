"""E04-X exploratory fixed-budget content substitution. Production code stays frozen."""
import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from types import MethodType

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('e03_driver',HERE.parent/'e03/run_pilot.py')
e3=importlib.util.module_from_spec(spec);spec.loader.exec_module(e3)
c=e3.c
PARENT=e3.PARENT
E02=HERE.parent/'e02/exploratory-20261008-01'
E03=HERE.parent/'e03/pilot-20261008-01'


def choose_pair(mapping,original):
    """No prediction, correctness or gold fields are used for selection."""
    if not mapping['eligible']:raise ValueError('Source evidence is not eligible')
    required=set(mapping['required_source_chunks'])
    if not required or not all(s['fully_covered'] for s in mapping['span_mapping']):
        raise ValueError('Incomplete source mapping')
    if len(original)!=2 or original!=sorted(original) or original[0]!=0:
        raise ValueError('Expected original canonical filler0 at position0')
    sources={s['slot_index']:s for s in mapping['slots']}
    filler=sources[0]
    if filler['direct_intersection'] or filler['indirect_possible_intersection']:
        raise ValueError('Filler has a recorded link to required sources')
    candidates=[i for i,s in sources.items() if i>0 and set(s['direct_intersection'])>=required]
    if not candidates:raise ValueError('No single direct source slot covers complete evidence')
    return [0,min(candidates)]


def prepare(output):
    if output.exists():raise ValueError('Output exists; no automatic overwrite/resume')
    e3.verify_inputs(c.read(E03/'manifest.json'))
    if c.read(E03/'status.json')['status']!='COMPLETE_DESCRIPTIVE_PILOT':
        raise ValueError('E03 not complete')
    prior=c.read(E03/'results.json')
    if not prior['repeat_stability_pass'] or prior['old_on_matches']!=10 or prior['old_on_selection_matches']!=10:
        raise ValueError('E03 reproducibility gate failed')
    for name,digest in c.read(E02/'acceptance.json')['core_sha256'].items():
        if c.sha(E02/name)!=digest:raise ValueError('E02 core changed')
    mappings=[r for r in e3.jsonl(E02/'source_slot_mapping.jsonl') if r['seed']==42 and r['eligible']]
    if len(mappings)!=6 or len({r['sample_id'] for r in mappings})!=6:
        raise ValueError('Expected six eligible samples without alternative-set duplication')
    inputs={str(p):c.sha(p) for p in [Path(__file__),HERE/'test_run_content.py',HERE/'test_tensor_adapter.py',
        HERE.parent/'E04_EXECUTION.md',E02/'acceptance.json',E02/'summary.json',
        E02/'source_slot_mapping.jsonl',E02/'adjudicated_ai_exploratory.jsonl',
        E03/'manifest.json',E03/'results.json',E03/'runtime.json',E03/'status.json']}
    plans=[]
    for m in mappings:
        ctx,q=m['context_index'],m['query_id']
        old={}
        for policy in ['full','random']:
            path=PARENT/f'jobs/s42-c{ctx}/queries/q{q:03d}_{policy}.json'
            old[policy]=e3.checked_envelope(path)['record'];inputs[str(path)]=c.sha(path)
            if (old[policy]['seed'],old[policy]['context_index'],old[policy]['query_id'],old[policy]['policy'])!=(42,ctx,q,policy):
                raise ValueError('Parent record identity mismatch')
            if old[policy]['selection'][0]['n']!=2 or old[policy]['selection'][0]['latent_count']!=16:
                raise ValueError('Parent budget mismatch')
        if old['full']['question']!=old['random']['question'] or old['full']['gold_answers']!=old['random']['gold_answers']:
            raise ValueError('Full/random question mismatch')
        original=old['full']['selection'][0]['selected_indices']
        related=choose_pair(m,original)
        plans.append(dict(sample_id=m['sample_id'],context_index=ctx,query_id=q,seed=42,
            original_indices=original,source_indices=related,
            random_indices=old['random']['selection'][0]['selected_indices'],
            source_chunks=m['required_source_chunks'],
            source_slot_version=next(s['slot_version'] for s in m['slots'] if s['slot_index']==related[1]),
            fixed_filler=0,fixed_replacement_position=1,semantic_negative=False))
    roster=e3.jsonl(e3.ROSTER)
    coverage=[dict(sample_id=r['sample_id'],included=any(p['sample_id']==r['sample_id'] for p in plans),
                   reason='AI source supported' if any(p['sample_id']==r['sample_id'] for p in plans)
                   else 'AI source conflict/ambiguous/not_locatable; not excluded by model output') for r in roster]
    manifest=dict(schema='E04-X/v1',created_at=c.now(),parent_contract=e3.CONTRACT,
        plans=plans,coverage_all10=coverage,max_new_generations=18,comparison_answers=18,
        original_and_random_reuse=True,primary='source-original all six strict difference',
        MEOI=None,formal_E02_pass=False,semantic_negative=False,efficacy_gate='UNDETERMINED',
        max_wall_seconds=3600,max_output_bytes=100*1024**2,input_sha256=inputs,
        paper_role='reference_only',section_id='memory_content',item_id='E04-X',
        claim_links=['C02','C05','C11','R04'])
    manifest['contract_sha256']=c.digest(manifest)
    output.mkdir(parents=True);c.atomic(output/'manifest.json',manifest)
    c.atomic(output/'status.json',dict(status='PREPARED',completed_generations=0))
    print(json.dumps(dict(status='PREPARED',plans=plans,max_new_generations=18),indent=2))


def verify_inputs(manifest):
    unsigned=dict(manifest);digest=unsigned.pop('contract_sha256')
    if c.digest(unsigned)!=digest:raise ValueError('Manifest changed')
    for path,digest in manifest['input_sha256'].items():
        if c.sha(path)!=digest:raise ValueError(f'Input changed: {path}')
    e3.verify_inputs(c.read(E03/'manifest.json'))


@contextmanager
def fixed_return(bank,indices,tensor_trace):
    """Wrap existing full selector, changing only returned slot identities/content.

    The existing read-only proxy restores bank access state. Native selection is
    recorded separately. No key/memory tensor in the bank is overwritten.
    """
    from scripts.eval.review_retrieval_policies import fingerprint
    old=bank.retrieve_with_context
    had_override='retrieve_with_context' in bank.__dict__
    if len(indices)!=2 or len(set(indices))!=2 or indices!=sorted(indices):
        raise ValueError('Need two distinct canonical slots')
    if any(i<0 or i>=len(bank._slots) for i in indices):raise ValueError('Slot missing')
    def retrieve(self,query_states,**kwargs):
        native=old(query_states,**kwargs)
        if len(native.slots)!=2:raise ValueError('Native full is not a two-slot retrieval')
        target=native.slots[0].memory
        selected=[];details=[]
        for i in indices:
            s=self._slots[i]
            if tuple(s.memory.shape)!=tuple(target.shape) or s.memory.shape[0]!=8:
                raise ValueError('Slot shape/token budget differs')
            memory=s.memory.to(device=target.device,dtype=target.dtype).detach().clone()
            key=s.key.to(device=target.device,dtype=target.dtype).detach().clone()
            expected=fingerprint(s.memory.to(dtype=target.dtype))
            actual=fingerprint(memory)
            if expected!=actual:raise ValueError('Returned tensor is not specified slot memory')
            details.append(dict(slot_index=i,memory_shape=list(memory.shape),memory_sha256=actual,
                expected_memory_sha256=expected,key_sha256=fingerprint(key),
                memory_norm=float(memory.float().norm().item()),key_norm=float(key.float().norm().item()),
                created_step=s.created_step,last_retrieved_step=s.last_retrieved_step))
            selected.append(replace(s,memory=memory,key=key,metadata=deepcopy(s.metadata)))
        tensor_trace.append(dict(actual_indices=list(indices),native_full_indices=list(native.retrieved_indices),
            actual_scores=[native.scores[i] for i in indices],tensor_details=details,
            forced_source_selection=(list(native.retrieved_indices)!=list(indices))))
        return replace(native,slots=selected,retrieved_indices=tuple(indices),
                       retrieved_scores=tuple(native.scores[i] for i in indices))
    bank.retrieve_with_context=MethodType(retrieve,bank)
    try:yield
    finally:
        if had_override:bank.retrieve_with_context=old
        else:del bank.retrieve_with_context


def worker(output,gpu):
    from scripts.eval.review_m2_smoke import eventqa,metrics,preflight,verify_query
    from scripts.eval.review_retrieval_policies import capture_bank,fingerprint,load_snapshot,policy_adapter,restore_bank,restore_rng
    import torch
    manifest=c.read(output/'manifest.json');started=time.monotonic();generated=0
    if c.read(output/'status.json')['status']!='PREPARED':raise ValueError('Already attempted; no auto-retry')
    def alarm(signum,frame):raise TimeoutError('60-minute execution cap reached')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(manifest['max_wall_seconds'])
    c.atomic(output/'status.json',dict(status='PREFLIGHT',pid=os.getpid(),gpu=gpu,completed_generations=0))
    try:
        verify_inputs(manifest);parent=c.read(PARENT/'manifest.json')
        args=eventqa.build_parser().parse_args([])
        for name,value in parent['protocol'].items():setattr(args,name,value)
        args.model_path=parent['model_path'];args.checkpoint_path=parent['checkpoint_path']
        args.seed=42;args.snapshot=None;args.output_root=str(output)
        check=preflight(args);c.atomic(output/'preflight.json',check);print('PREFLIGHT',check,flush=True)
        if (check['gate']!='PASS' or check.get('gpu_name')!='NVIDIA RTX A6000' or
            check.get('free_vram_bytes',0)<12*1024**3 or torch.cuda.device_count()!=1):
            raise RuntimeError('GPU/path/topology gate failed; no model loaded')
        rows=eventqa._load_rows(args.parquet,eventqa.SUB_DATASET)
        if len(rows)!=5:raise ValueError('Wrong dataset')
        model,capacity=eventqa.weaver_bank._load_model(args)
        runtime=eventqa._runtime_reproducibility_metadata(args,selected_context_indices=list(range(5)),model=model)
        c.atomic(output/'runtime.json',runtime)
        environment=c.read(PARENT/'environment_contract.json')
        if {k:runtime[k] for k in environment}!=environment:raise ValueError('Runtime differs')
        config=eventqa._eventqa_bank_config(args);contexts={};snapshots={}
        print(f'WORKER_READY gpu={gpu} pid={os.getpid()} max_new=18',flush=True)
        for plan in manifest['plans']:
            ctx,q=plan['context_index'],plan['query_id'];args.context_index=ctx
            if ctx not in contexts:
                contexts[ctx]=eventqa.build_context_payload(args,rows[ctx],ctx,parent['created_at'])
                prov=c.read(PARENT/f'jobs/s42-c{ctx}/context_identity.json')
                if [metrics.text_sha(s) for s in contexts[ctx]['chunks']]!=prov['chunk_sha256']:
                    raise ValueError('Chunk drift')
                snapshots[ctx]=load_snapshot(PARENT/f'jobs/s42-c{ctx}/frozen_bank.pt')
                if runtime['deterministic_state']!=c.read(PARENT/f'jobs/s42-c{ctx}/runtime.json')['runtime']['deterministic_state']:
                    raise ValueError('Determinism drift')
            saved=snapshots[ctx]
            if any(tuple(s.memory.shape)!=(8,1536) for s in saved['bank']['_slots']):
                raise ValueError('Expected eight Weaver-space tokens per slot')
            old=e3.checked_envelope(PARENT/f'jobs/s42-c{ctx}/queries/q{q:03d}_full.json')
            question=eventqa.build_question_payload(contexts[ctx],q);payload=eventqa._query_only_payload(question)
            if question['question']!=old['record']['question'] or metrics.text_sha(payload['query_prompt'])!=old['record']['query_sha256']:
                raise ValueError('Question/prompt drift')
            related_records=[]
            for arm,repeat in [('original_replay',0),('source',0),('source',1)]:
                indices=plan['original_indices'] if arm=='original_replay' else plan['source_indices']
                bank=restore_bank(saved['bank']);restore_rng(saved['rng'])
                before=fingerprint(capture_bank(bank));trace=[];tensor_trace=[]
                active=dict(sample_id=plan['sample_id'],arm=arm,repeat=repeat,indices=indices)
                c.atomic(output/'status.json',dict(status='RUNNING',pid=os.getpid(),gpu=gpu,
                    completed_generations=generated,active_task=active))
                print('QUERY_START',active,flush=True)
                with policy_adapter(bank,'full',last_write=saved['last_write'],seed=42,context=ctx,query=q,trace=trace):
                    with fixed_return(bank,indices,tensor_trace):
                        result=eventqa._run_eventqa_model(args,model,capacity,payload,'on',config,
                            external_bank=bank,preserve_bank=True,recorded_bank_config=config)
                result.pop('_retained_bank',None)
                verify_query(result,trace,before,bank);e3.validate_generation(result,trace,'on')
                if len(tensor_trace)!=1 or result['generations'][-1]['retrieved_indices']!=indices:
                    raise ValueError('Returned indices or retrieval count differ')
                if result['rendered_query_prompt']!=old['runtime_result']['rendered_query_prompt']:
                    raise ValueError('Rendered prompt differs')
                record=dict(sample_id=plan['sample_id'],context_index=ctx,query_id=q,seed=42,arm=arm,repeat=repeat,
                    question=question['question'],gold_answers=question['gold_answers'],prediction=result['prediction'],
                    native_full_selection=trace,actual_indices=indices,tensor_trace=tensor_trace,
                    format_flags=eventqa._format_flags(result['prediction']),
                    rendered_prompt_sha256=metrics.text_sha(result['rendered_query_prompt']),
                    snapshot_unchanged=True,query_write_count_delta=0,
                    latency_seconds=result['latency_seconds'],peak_cuda_memory=result['peak_cuda_memory'])
                envelope=dict(record=record,runtime_result=result,contract_sha256=manifest['contract_sha256'])
                envelope['content_sha256']=c.digest(envelope)
                path=output/'queries'/f'{plan["sample_id"]}_{arm}_r{repeat}.json'
                if path.exists():raise ValueError('Existing answer must not be overwritten')
                c.atomic(path,envelope);generated+=1;bank.reset()
                print(f'QUERY_COMPLETE {plan["sample_id"]} {arm} r{repeat} {generated}/18',flush=True)
                if trace!=old['record']['selection']:raise ValueError('Native retrieval differs from parent')
                if arm=='original_replay' and record['prediction']!=old['record']['prediction']:
                    raise ValueError('Original replay differs; no comparator reuse permitted')
                if arm=='source':
                    related_records.append(record)
                    if repeat==1 and (record['prediction']!=related_records[0]['prediction'] or
                            record['tensor_trace']!=related_records[0]['tensor_trace']):
                        raise ValueError('Source repeat differs; preserve and stop')
                if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>manifest['max_output_bytes']:
                    raise RuntimeError('Output cap exceeded')
        if generated!=18:raise ValueError('Wrong count')
        verify_inputs(manifest)
        c.atomic(output/'status.json',dict(status='GENERATED',pid=os.getpid(),gpu=gpu,
            completed_generations=generated,elapsed_seconds=time.monotonic()-started,finished_at=c.now()))
        subprocess.run([c.SCORER_PYTHON,str(Path(__file__)),'score','--output',str(output)],cwd=ROOT,check=True)
        print('PILOT_COMPLETE',str(output),flush=True)
    except BaseException as exc:
        c.atomic(output/'status.json',dict(status='STOPPED',completed_generations=generated,pid=os.getpid(),gpu=gpu,
            error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),finished_at=c.now()))
        raise
    finally:signal.alarm(0)


def paired(source,other):
    if set(source)!=set(other):raise ValueError('Paired roster differs')
    keys=sorted(source)
    def transitions(field):
        counts={f'source_{a}_other_{b}':0 for a in [0,1] for b in [0,1]}
        for k in keys:counts[f'source_{int(bool(source[k][field]))}_other_{int(bool(other[k][field]))}']+=1
        return counts
    def state(r):return 'C' if r['strict_candidate_correct'] else 'W' if r['candidate_valid'] else 'I'
    cwi={a+'->'+b:0 for a in 'CWI' for b in 'CWI'}
    for k in keys:cwi[state(other[k])+'->'+state(source[k])]+=1
    contexts=sorted({source[k]['context_index'] for k in keys})
    delta={str(ctx):sum(int(source[k]['strict_candidate_correct'])-int(other[k]['strict_candidate_correct'])
        for k in keys if source[k]['context_index']==ctx)/sum(source[k]['context_index']==ctx for k in keys)
        for ctx in contexts}
    return dict(denominator=len(keys),strict=transitions('strict_candidate_correct'),
        parser_nonempty=transitions('parsed_nonempty'),candidate_valid=transitions('candidate_valid'),
        raw_gold_any=transitions('raw_gold_any'),raw_recall_positive=transitions('official_raw_recall'),
        other_to_source_CWI=cwi,text_changed=sum(source[k]['prediction']!=other[k]['prediction'] for k in keys),
        context_deltas=delta,context_equal_mean=sum(delta.values())/len(delta),
        leave_one_context_out={k:sum(v for ctx,v in delta.items() if ctx!=k)/(len(delta)-1) for k in delta},
        significance='NOT_TESTED',semantic_relatedness_causality='NOT_ESTABLISHED')


def score(output):
    from scripts.eval import review_eventqa_decomposition as m
    if c.read(output/'status.json')['status']!='GENERATED':raise ValueError('Not fully generated')
    manifest=c.read(output/'manifest.json');verify_inputs(manifest);official=m.load_official()
    records=[];groups={arm:{} for arm in ['original','source','random']}
    files=list((output/'queries').glob('*.json'))
    if len(files)!=18:raise ValueError('Expected eighteen new generation envelopes')
    for plan in manifest['plans']:
        sid=plan['sample_id'];ctx,q=plan['context_index'],plan['query_id']
        for arm in groups:
            if arm=='source':
                e=c.read(output/'queries'/f'{sid}_source_r0.json');u=dict(e);digest=u.pop('content_sha256')
                if c.digest(u)!=digest:raise ValueError('New answer checksum differs')
                r=deepcopy(e['record']);r['origin']='new_source_repeat0'
            else:
                policy='full' if arm=='original' else 'random'
                e=e3.checked_envelope(PARENT/f'jobs/s42-c{ctx}/queries/q{q:03d}_{policy}.json')
                r=deepcopy(e['record']);r.update(sample_id=sid,arm=arm,origin='reused_M3',
                    actual_indices=r['selection'][0]['selected_indices'])
            info=m.candidate_info(r['question'],r['gold_answers'],official)
            if info['errors']:raise ValueError(info['errors'])
            r.update(m.evaluate(r['prediction'],r['gold_answers'],info,r['format_flags'],official))
            records.append(r);groups[arm][sid]=r
    summaries={arm:m.summarize([dict(r,official_em=r['official_em'] or 0,
        official_raw_recall=r['official_raw_recall'] or 0) for r in rows.values()]) for arm,rows in groups.items()}
    result=dict(status='COMPLETE_EXPLORATORY_CONTENT_PILOT',records=records,summary=summaries,
        paired_source_original=paired(groups['source'],groups['original']),
        paired_source_random_auxiliary=paired(groups['source'],groups['random']),
        primary_answer_count=18,new_generation_count=18,source_repeat_stability=True,
        original_replay_matches=6,coverage_all10=manifest['coverage_all10'],
        MEOI=None,formal_E02_pass=False,semantic_negative=False,efficacy_gate='UNDETERMINED',
        claim_boundary='latent identity/value sensitivity only; source semantics and fact retention unproven',
        scorer_sha256=c.sha(m.OFFICIAL),contract_sha256=manifest['contract_sha256'],
        next_route='STOP; review pilot before any expansion or algorithm change')
    if (output/'results.json').exists():raise ValueError('Results already exist')
    c.atomic(output/'results.json',result)
    status=c.read(output/'status.json');status.update(status=result['status'],scored_at=c.now())
    c.atomic(output/'status.json',status)
    print(json.dumps(dict(status=result['status'],paired=result['paired_source_original'],
        summary={k:v['strict_candidate_accuracy'] for k,v in summaries.items()}),indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','worker','score'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',default='5');a=p.parse_args()
    if a.action=='prepare':prepare(a.output.resolve())
    elif a.action=='worker':worker(a.output.resolve(),a.gpu)
    else:score(a.output.resolve())


if __name__=='__main__':main()
