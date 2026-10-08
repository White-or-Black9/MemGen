"""Bounded E03 replay using existing M3 interfaces; no algorithm changes."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.eval import review_m3_formal as c

HERE = Path(__file__).resolve().parent
PARENT = ROOT/'outputs/mab/review_m3_formal/20261002-m3-six-policy-v3'
ROSTER = HERE.parent/'e02/prepare-20261008-01/coordinator_only/roster_and_gold.jsonl'
PUBLIC = HERE.parent/'e02/prepare-20261008-01/rater_A/questions.jsonl'
CONTRACT = 'a548306d4131e4df0cba1340d3272a4882feae1d9c993c881e281f4551e433c9'


def jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def checked_envelope(path):
    e=c.read(path); unsigned=dict(e); digest=unsigned.pop('content_sha256')
    if c.digest(unsigned)!=digest or e['complete'] is not True:
        raise ValueError(f'Invalid parent envelope: {path}')
    return e


def tasks(roster):
    if len(roster)!=10 or len({r['sample_id'] for r in roster})!=10:
        raise ValueError('Expected ten unique frozen samples')
    if Counter(r['context_index'] for r in roster)!=Counter({i:2 for i in range(5)}):
        raise ValueError('Expected two samples/context')
    if len({(r['context_index'],r['query_id']) for r in roster})!=10:
        raise ValueError('Duplicate question')
    return [dict(sample_id=r['sample_id'],context_index=r['context_index'],query_id=r['query_id'],
                 arm=arm,repeat=repeat,seed=42)
            for r in roster for repeat in range(2) for arm in ['on','off']]


def prepare(output):
    if output.exists():raise ValueError('Output exists; never overwrite/restart automatically')
    parent=c.read(PARENT/'manifest.json')
    if parent['contract_sha256']!=CONTRACT:raise ValueError('Wrong M3 contract')
    c.verify_manifest(parent,check_inputs=True)
    roster=jsonl(ROSTER); tasklist=tasks(roster)
    public={r['sample_id']:r for r in jsonl(PUBLIC)}
    inputs={str(p):c.sha(p) for p in [PARENT/'manifest.json',PARENT/'results.json',
            PARENT/'environment_contract.json',ROSTER,PUBLIC,Path(__file__),HERE/'test_run_pilot.py',
            HERE.parent/'E03_EXECUTION.md']}
    for ctx in range(5):
        job=PARENT/f'jobs/s42-c{ctx}'
        identity=c.read(job/'snapshot_identity.json')
        if c.sha(job/'frozen_bank.pt')!=identity['snapshot_file_sha256']:
            raise ValueError('Snapshot changed')
        for name in ['frozen_bank.pt','snapshot_identity.json','context_identity.json','runtime.json']:
            inputs[str(job/name)]=c.sha(job/name)
    for r in roster:
        path=PARENT/f"jobs/s42-c{r['context_index']}/queries/q{r['query_id']:03d}_full.json"
        e=checked_envelope(path); old=e['record']; q=public[r['sample_id']]
        if (old['seed'],old['context_index'],old['query_id'],old['policy'])!=(42,r['context_index'],r['query_id'],'full'):
            raise ValueError('Parent question identity differs')
        if c.digest(old['question'])!=c.digest(q['question']) or old['gold_answers']!=r['gold_answers']:
            raise ValueError('Public question/parent gold differs')
        if hashlib.sha256(q['question'].encode()).hexdigest()!=r['question_sha256']:
            raise ValueError('Question hash differs')
        inputs[str(path)]=c.sha(path)
    manifest=dict(schema='M6-E03-pilot/v1',parent_contract=CONTRACT,parent=str(PARENT),
        created_at=c.now(),tasks=tasklist,expected_generations=40,seed=42,
        primary='all ten questions strict on-off using repeat0 only',MEOI=None,
        efficacy_gate='NOT_APPROVED; descriptive pilot only',max_wall_seconds=7200,
        max_output_bytes=100*1024**2,input_sha256=inputs,protocol=parent['protocol'],
        model_path=parent['model_path'],checkpoint_path=parent['checkpoint_path'],
        source_sha256=parent['source_sha256'],parent_input_sha256=parent['input_sha256'],
        paper_role='main_required_candidate',section_id='conditioning',item_id='E03',
        claim_links=['C05','R01','R05','R06'])
    manifest['git_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    manifest['contract_sha256']=c.digest(manifest)
    output.mkdir(parents=True)
    c.atomic(output/'manifest.json',manifest)
    c.atomic(output/'status.json',dict(status='PREPARED',completed_generations=0,new_gpu_generations=0))
    print(json.dumps(dict(status='PREPARED',output=str(output),tasks=len(tasklist)),indent=2))


def verify_inputs(manifest):
    unsigned=dict(manifest); digest=unsigned.pop('contract_sha256')
    if c.digest(unsigned)!=digest:raise ValueError('E03 manifest changed')
    for path,expected in manifest['input_sha256'].items():
        if c.sha(path)!=expected:raise ValueError(f'Input drift: {path}')
    c.verify_manifest(c.read(PARENT/'manifest.json'),check_inputs=True)


def validate_generation(result,trace,arm):
    if len(trace)!=1 or trace[0]['n']!=2 or trace[0]['latent_count']!=16:
        raise ValueError('Retrieval budget mismatch')
    g=result['generations'][-1]
    if (g['query_weaver_invoke_count']!=1 or not g['weaver_output_generated'] or
        not g['weaver_output_consumed'] or g['direct_injection_applied']):
        raise ValueError('Wrong regeneration path')
    if g['weaver_conditioning_token_count']!=(16 if arm=='on' else 0):
        raise ValueError('Wrong conditioning budget')
    if bool(g['weaver_conditioned_on_retrieved_memory'])!=(arm=='on'):
        raise ValueError('Wrong conditioning flag')
    if result['context_memorization_performed'] or result['construction_turn_diagnostics']:
        raise ValueError('Unexpected bank construction')
    if result['effective_generation_max_length']!=40:
        raise ValueError('Decoding cap changed')


def same_replay(a,b):
    return (a['prediction']==b['prediction'] and a['selection']==b['selection'] and
            a['rendered_prompt_sha256']==b['rendered_prompt_sha256'])


def worker(output,gpu):
    # Lazy model imports: prepare, tests and scoring do not load torch/model.
    from scripts.eval.review_m2_smoke import eventqa,metrics,preflight,verify_query
    from scripts.eval.review_retrieval_policies import (
        capture_bank,fingerprint,load_snapshot,policy_adapter,restore_bank,restore_rng)
    import torch
    manifest=c.read(output/'manifest.json'); generated=0; started=time.monotonic()
    if c.read(output/'status.json')['status']!='PREPARED':
        raise ValueError('Worker already attempted; do not auto-retry')
    def timed_out(signum,frame):raise TimeoutError('E03 120-minute wall cap reached')
    signal.signal(signal.SIGALRM,timed_out);signal.alarm(manifest['max_wall_seconds'])
    c.atomic(output/'status.json',dict(status='PREFLIGHT',pid=os.getpid(),gpu=gpu,
                                      completed_generations=0,started_at=c.now()))
    try:
        verify_inputs(manifest)
        args=eventqa.build_parser().parse_args([])
        for name,value in manifest['protocol'].items():setattr(args,name,value)
        args.model_path=manifest['model_path'];args.checkpoint_path=manifest['checkpoint_path']
        args.seed=42;args.snapshot=None;args.output_root=str(output)
        check=preflight(args);c.atomic(output/'preflight.json',check);print('PREFLIGHT',check,flush=True)
        if (check['gate']!='PASS' or check.get('gpu_name')!='NVIDIA RTX A6000' or
            check.get('free_vram_bytes',0)<12*1024**3 or torch.cuda.device_count()!=1):
            raise RuntimeError('GPU/model path/topology resource gate failed; no model loaded')
        rows=eventqa._load_rows(args.parquet,eventqa.SUB_DATASET)
        if len(rows)!=5:raise ValueError('Wrong dataset contexts')
        config=eventqa._eventqa_bank_config(args)
        model,capacity=eventqa.weaver_bank._load_model(args)
        runtime=eventqa._runtime_reproducibility_metadata(args,selected_context_indices=list(range(5)),model=model)
        environment={k:runtime[k] for k in c.read(PARENT/'environment_contract.json')}
        c.atomic(output/'runtime.json',runtime)
        if environment!=c.read(PARENT/'environment_contract.json'):
            raise ValueError('M3 runtime environment differs')
        public={r['sample_id']:r for r in jsonl(PUBLIC)}
        records={}; contexts={}; snapshots={}
        print(f'WORKER_READY gpu={gpu} pid={os.getpid()} planned=40',flush=True)
        for task in manifest['tasks']:
            ctx=task['context_index']; q=task['query_id'];arm=task['arm'];repeat=task['repeat']
            args.context_index=ctx
            if ctx not in contexts:
                contexts[ctx]=eventqa.build_context_payload(args,rows[ctx],ctx,c.read(PARENT/'manifest.json')['created_at'])
                oldcontext=c.read(PARENT/f'jobs/s42-c{ctx}/context_identity.json')
                if [metrics.text_sha(chunk) for chunk in contexts[ctx]['chunks']]!=oldcontext['chunk_sha256']:
                    raise ValueError('Chunk identity drift')
                snapshots[ctx]=load_snapshot(PARENT/f'jobs/s42-c{ctx}/frozen_bank.pt')
                olddet=c.read(PARENT/f'jobs/s42-c{ctx}/runtime.json')['runtime']['deterministic_state']
                if runtime['deterministic_state']!=olddet:raise ValueError('Determinism settings differ')
            saved=snapshots[ctx]
            question=eventqa.build_question_payload(contexts[ctx],q)
            old=checked_envelope(PARENT/f'jobs/s42-c{ctx}/queries/q{q:03d}_full.json')
            if question['question']!=public[task['sample_id']]['question']:
                raise ValueError('Frozen question differs')
            payload=eventqa._query_only_payload(question)
            if metrics.text_sha(payload['query_prompt'])!=old['record']['query_sha256']:
                raise ValueError('Query user prompt differs')
            bank=restore_bank(saved['bank']);restore_rng(saved['rng'])
            before=fingerprint(capture_bank(bank));trace=[]
            c.atomic(output/'status.json',dict(status='RUNNING',pid=os.getpid(),gpu=gpu,
                completed_generations=generated,active_task=task,started_at=c.now()))
            print('QUERY_START',task,flush=True)
            with policy_adapter(bank,'full',last_write=saved['last_write'],seed=42,context=ctx,query=q,trace=trace):
                result=eventqa._run_eventqa_model(args,model,capacity,payload,'on',config,
                    external_bank=bank,preserve_bank=True,recorded_bank_config=config,
                    query_retrieved_memory_conditioning=(arm=='on'),
                    query_latent_usage='weaver_integrated' if arm=='on' else 'retrieve_but_do_not_condition')
            result.pop('_retained_bank',None)
            verify_query(result,trace,before,bank);validate_generation(result,trace,arm)
            if result['rendered_query_prompt']!=old['runtime_result']['rendered_query_prompt']:
                raise ValueError('Rendered prompt differs')
            record=dict(**task,policy=arm,question=question['question'],gold_answers=question['gold_answers'],
                prediction=result['prediction'],selection=trace,format_flags=eventqa._format_flags(result['prediction']),
                rendered_prompt_sha256=metrics.text_sha(result['rendered_query_prompt']),
                query_write_count_delta=0,snapshot_unchanged=True,
                old_on_prediction_equal=(result['prediction']==old['record']['prediction']) if arm=='on' else None,
                old_on_selection_equal=(trace==old['record']['selection']),
                latency_seconds=result['latency_seconds'],peak_cuda_memory=result['peak_cuda_memory'])
            path=output/'queries'/f'{task["sample_id"]}_{arm}_r{repeat}.json'
            if path.exists():raise ValueError('Existing query must not be overwritten')
            envelope=dict(record=record,runtime_result=result,contract_sha256=manifest['contract_sha256'])
            envelope['content_sha256']=c.digest(envelope);c.atomic(path,envelope)
            generated+=1;records[(ctx,q,arm,repeat)]=record;bank.reset()
            print(f'QUERY_COMPLETE {task["sample_id"]} {arm} r{repeat} completed={generated}/40',flush=True)
            on=records.get((ctx,q,'on',repeat))
            if arm=='off' and trace!=on['selection']:raise ValueError('On/off retrieval differs')
            if repeat==1 and not same_replay(record,records[(ctx,q,arm,0)]):
                raise ValueError('Current repeated prediction/selection differs; stop, preserve both')
            if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>manifest['max_output_bytes']:
                raise RuntimeError('Output cap exceeded')
        verify_inputs(manifest)
        if generated!=40:raise ValueError('Generation count differs')
        c.atomic(output/'status.json',dict(status='GENERATED',completed_generations=40,pid=os.getpid(),
            gpu=gpu,elapsed_seconds=time.monotonic()-started,finished_at=c.now()))
        subprocess.run([c.SCORER_PYTHON,str(Path(__file__)),'score','--output',str(output)],cwd=ROOT,check=True)
        print('PILOT_COMPLETE',str(output),flush=True)
    except BaseException as exc:
        c.atomic(output/'status.json',dict(status='STOPPED',completed_generations=generated,pid=os.getpid(),
            gpu=gpu,error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),finished_at=c.now()))
        raise
    finally:signal.alarm(0)


def pair_tables(on,off):
    keys=sorted(on)
    if keys!=sorted(off):raise ValueError('Paired question mismatch')
    def transitions(field):
        return dict(Counter(f'on_{int(bool(on[k][field]))}_off_{int(bool(off[k][field]))}' for k in keys))
    def state(row):return 'C' if row['strict_candidate_correct'] else 'W' if row['candidate_valid'] else 'I'
    cwi={a+'->'+b:0 for a in 'CWI' for b in 'CWI'}
    for k in keys:cwi[state(off[k])+'->'+state(on[k])]+=1
    context={str(ctx):sum(int(on[k]['strict_candidate_correct'])-int(off[k]['strict_candidate_correct'])
                  for k in keys if k[0]==ctx)/sum(k[0]==ctx for k in keys) for ctx in range(5)}
    return dict(strict_correctness=transitions('strict_candidate_correct'),
        parser_nonempty=transitions('parsed_nonempty'),candidate_valid=transitions('candidate_valid'),
        raw_gold_any=transitions('raw_gold_any'),raw_recall_positive=transitions('official_raw_recall'),
        off_to_on_CWI=cwi,context_deltas=context,context_equal_mean=sum(context.values())/5,
        leave_one_context_out={ctx:sum(v for k,v in context.items() if k!=ctx)/4 for ctx in context},
        denominator=len(keys),significance='NOT_TESTED: at most five dependent-source clusters')


def score(output):
    from scripts.eval import review_eventqa_decomposition as m
    if c.read(output/'status.json')['status']!='GENERATED':raise ValueError('Not fully generated')
    manifest=c.read(output/'manifest.json');verify_inputs(manifest)
    official=m.load_official();records=[]
    paths=list((output/'queries').glob('*.json'))
    if len(paths)!=40:raise ValueError('Expected exactly forty records')
    for path in paths:
        e=c.read(path);unsigned=dict(e);digest=unsigned.pop('content_sha256')
        if c.digest(unsigned)!=digest:raise ValueError('Query record checksum differs')
        r=e['record'];info=m.candidate_info(r['question'],r['gold_answers'],official)
        if info['errors']:raise ValueError(info['errors'])
        r.update(m.evaluate(r['prediction'],r['gold_answers'],info,r['format_flags'],official))
        records.append(r)
    primary=[r for r in records if r['repeat']==0]
    on={(r['context_index'],r['query_id']):r for r in primary if r['arm']=='on'}
    off={(r['context_index'],r['query_id']):r for r in primary if r['arm']=='off'}
    if len(on)!=10 or len(off)!=10:raise ValueError('Primary roster incomplete')
    summaries={arm:m.summarize([dict(r,official_em=r['official_em'] or 0,
                            official_raw_recall=r['official_raw_recall'] or 0)
                  for r in primary if r['arm']==arm]) for arm in ['on','off']}
    result=dict(status='COMPLETE_DESCRIPTIVE_PILOT',records=records,summary=summaries,
        paired=pair_tables(on,off),primary_repeat=0,repeat1_role='reproducibility_only_not_extra_samples',
        repeat_stability_pass=True,old_on_matches=sum(r['old_on_prediction_equal'] for r in primary if r['arm']=='on'),
        old_on_selection_matches=sum(r['old_on_selection_equal'] for r in primary if r['arm']=='on'),
        MEOI=None,efficacy_gate='UNDETERMINED',gate_research_authorized=False,
        next_route='STOP: review repeat stability and paired effects before any expansion',
        scorer_sha256=c.sha(m.OFFICIAL),contract_sha256=manifest['contract_sha256'])
    if (output/'results.json').exists():raise ValueError('Results already exist')
    c.atomic(output/'results.json',result)
    status=c.read(output/'status.json');status.update(status='COMPLETE_DESCRIPTIVE_PILOT',scored_at=c.now())
    c.atomic(output/'status.json',status)
    print(json.dumps(dict(status=result['status'],paired=result['paired'],old_on_matches=result['old_on_matches']),indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','worker','score'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',default='0');a=p.parse_args()
    if a.action=='prepare':prepare(a.output.resolve())
    elif a.action=='worker':worker(a.output.resolve(),a.gpu)
    else:score(a.output.resolve())


if __name__=='__main__':main()
