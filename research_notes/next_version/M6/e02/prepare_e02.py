"""Deterministic, model-free export of an unannotated independent-rater pilot package."""
import argparse
import ast
import collections
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'e01'))
from audit_e01 import ROOT, CONTRACT, atomic, inventory, jsonl, read, require, sha, token_spans
PARENT=ROOT/'outputs/mab/review_m3_formal/20261002-m3-six-policy-v3'
M4=HERE.parents[1]/'m4'
BANNED={'gold_answers','gold_candidate_index','model_answer','prediction','policy','scores',
        'selected_indices','slot_index','slot_version','query_id','lexical_matches'}
STATUSES={'supported','ambiguous','not_locatable','conflict','insufficient_context','prior_only'}


def text_sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def lines(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f]


def select_pilot(rows):
    require(len(rows)==50 and len({(r['context_index'],r['query_id']) for r in rows})==50,'blind50 identity')
    chosen=[]
    for ctx in range(5):
        group=[r for r in rows if r['context_index']==ctx]
        require(len(group)==10,'blind sample balance')
        chosen.extend(sorted(group,key=lambda r:text_sha(f"M4-A-20261008|{ctx}|{r['query_id']}"))[:2])
    return chosen


def candidates(question):
    a,b,_=next(x for x in token_spans(question,question) if x[2]=='candidate_list')
    result=ast.literal_eval(question[a:b])
    require(isinstance(result,list) and len(result)>1 and all(isinstance(x,str) for x in result),'invalid candidates')
    return result


def public_question(row,sample_id,context_sha):
    return dict(sample_id=sample_id,context_index=row['context_index'],context_id=row['context_id'],
                context_file=f"contexts/c{row['context_index']}.txt",context_sha256=context_sha,
                question=row['question'],question_sha256=row['question_text_sha256'],
                candidates=candidates(row['question']),candidate_index_base=0)


def blank_form(q):
    return dict(schema='E02-phase1/v1',sample_id=q['sample_id'],rater_id=None,
                context_sha256=q['context_sha256'],question_sha256=q['question_sha256'],
                prior_exposure=None,status=None,chosen_candidate_indices=[],evidence_sets=[],
                notes=None,elapsed_minutes=None,completed_at=None)


def check_blind(value):
    if isinstance(value,dict):
        require(not BANNED.intersection(value),'initial rater package contains restricted fields')
        for v in value.values():check_blind(v)
    elif isinstance(value,list):
        for v in value:check_blind(v)


def check_blank(form):
    require(form['status'] is None and form['rater_id'] is None and not form['evidence_sets']
            and not form['chosen_candidate_indices'] and form['completed_at'] is None,'prefilled human label')


def check_span(context,span):
    a,b=span['start'],span['end']
    require(type(a) is int and type(b) is int and 0<=a<b<=len(context),'bad Unicode span offsets')
    require(context[a:b]==span['text'] and bool(span['text'].strip()),'span exact text mismatch')
    return True


def write_text(path,value):
    # Generated dataset copies / human-readable package artifacts, never edits parent files.
    with Path(path).open('x',encoding='utf-8',newline='') as f:f.write(value)


def verify_sources():
    receipt=read(M4/'acceptance.json')
    require(receipt['status']=='PASS','M4 not accepted')
    for path in receipt['compared_runs']:
        for name,h in receipt['identical_artifact_sha256'].items():
            require(sha(ROOT/path/name)==h,'M4 core drift: '+name)
    manifest=read(PARENT/'manifest.json')
    require(manifest['contract_sha256']==CONTRACT,'wrong M3 contract')
    reference=read(HERE.parent/'e01/run-20261008-01/manifest.json')['parent_manifest_sha256']
    require(sha(PARENT/'manifest.json')==reference,'M3 manifest drift')
    parquet=Path(next(p for p in manifest['input_sha256'] if p.endswith('.parquet')))
    require(sha(parquet)==manifest['input_sha256'][str(parquet)],'dataset drift')
    return manifest,parquet


def run(output):
    start=time.monotonic()
    require(not output.exists() and output.parent==HERE,'fresh output inside e02 required')
    manifest,parquet=verify_sources()
    before={str(p):inventory(p) for p in (PARENT,M4,HERE.parent/'e01',HERE.parent/'e01v')}
    blind=M4/'run-20261008-a1/blind_sample.jsonl'
    pilot=select_pilot(lines(blind))
    import pyarrow.parquet as pq
    dataset=[r for r in pq.read_table(parquet).to_pylist() if r['metadata']['source']=='eventqa_65536']
    require(len(dataset)==5,'context source scope')
    contexts={};identities={}
    for ctx,row in enumerate(dataset):
        text=row['context'];identity=read(PARENT/'jobs'/f's42-c{ctx}'/'context_identity.json')
        require('eventqa-'+text_sha(text)[:16]==identity['context_id'],'context identity mismatch')
        contexts[ctx]=text;identities[ctx]=identity
    questions=[];coordinator=[];forms=[]
    for i,row in enumerate(pilot):
        ctx,qid=row['context_index'],row['query_id'];original=dataset[ctx]
        require(row['question']==original['questions'][qid] and row['gold_answers']==original['answers'][qid], 'frozen QA drift')
        require(text_sha(row['question'])==row['question_text_sha256'],'question checksum')
        sample_id=f'E02P-c{ctx}-{i%2+1:02d}'
        q=public_question(row,sample_id,text_sha(contexts[ctx]));f=blank_form(q)
        check_blind(q);check_blind(f);check_blank(f)
        questions.append(q);forms.append(f)
        coordinator.append(dict(sample_id=sample_id,context_index=ctx,query_id=qid,
                                qa_pair_id=row['qa_pair_id'],gold_answers=row['gold_answers'],
                                context_id=row['context_id'],question_sha256=row['question_text_sha256']))
    chunks=lines(M4/'run-20261008-a1/chunk_offsets.jsonl')
    for chunk in chunks:
        text=contexts[chunk['context_index']]
        sentence_text=' '.join(text[a:b] for a,b in chunk['sentence_spans'])
        require(text_sha(sentence_text)==chunk['chunk_sha256'],'sentence-source chunk hash drift')
        require(chunk['chunk_sha256']==identities[chunk['context_index']]['chunk_sha256'][chunk['chunk_index']],'chunk identity drift')
    sources=lines(M4/'run-20261008-a1/provenance.jsonl')
    keep=('job_id','seed','context_index','slot_index','slot_version','direct_source_chunk',
          'direct_source_chunk_sha256','indirect_possible_source_chunks','potential_source_chunks',
          'previous_content_version','prior_retrieved_versions','current_write_turn','refresh_count')
    source_index=[{k:r[k] for k in keep} for r in sources]
    output.mkdir();atomic(output/'status.json',dict(status='PREPARING',human_annotations=0))
    for role in ('rater_A','rater_B'):
        packet=output/role;packet.mkdir();(packet/'contexts').mkdir()
        write_text(packet/'README.md',(HERE/'RATER_GUIDE.md').read_text())
        for ctx,text in contexts.items():write_text(packet/'contexts'/f'c{ctx}.txt',text)
        jsonl(packet/'questions.jsonl',questions)
        jsonl(packet/'annotations_blank.jsonl',forms)
    control=output/'coordinator_only';control.mkdir()
    jsonl(control/'roster_and_gold.jsonl',coordinator)
    jsonl(control/'chunk_offsets.jsonl',chunks)
    jsonl(control/'slot_sources.jsonl',source_index)
    jsonl(control/'phase2_gold_review_blank.jsonl',[
        dict(sample_id=q['sample_id'],rater_id=None,phase1_file_sha256=None,
             self_gold_agreement=None,suspected_label_issue=None,notes=None) for q in questions])
    jsonl(control/'adjudication_blank.jsonl',[
        dict(sample_id=q['sample_id'],adjudicator_id=None,phase1_A_sha256=None,phase1_B_sha256=None,
             adjudicated_status=None,accepted_evidence_sets=[],reason=None,unresolved=None) for q in questions])
    atomic(control/'release_gate.json',dict(status='NOT_RELEASED',rater_A=None,rater_B=None,adjudicator=None,
        training_and_budget_approved=False,prior_exposure_declarations=None,
        approved_thresholds=dict(coverage=None,status_agreement=None,span_agreement=None,
                                 fact_agreement=None,related_unrelated_candidate_availability=None),
        approval_record=None,annotation_status='NOT_STARTED',human_annotations=0))
    outputs={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='status.json'}
    require(before=={str(p):inventory(p) for p in (PARENT,M4,HERE.parent/'e01',HERE.parent/'e01v')},'parent inventory changed')
    verify_sources()
    require('torch' not in sys.modules and 'transformers' not in sys.modules,'model dependency imported')
    elapsed=time.monotonic()-start;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    require(elapsed<600 and rss<=4*1024**2,'resource budget exceeded')
    atomic(output/'manifest.json',dict(schema='M6-E02-PREP/v1',status='PREPARATION_COMPLETE',
        annotation_status='NOT_STARTED',human_annotations=0,new_model_runs=0,gpu_used=False,
        parent_contract_sha256=CONTRACT,parent_manifest_sha256=sha(PARENT/'manifest.json'),
        parent_blind50_sha256=sha(blind),dataset_sha256=sha(parquet),
        parent_preserved=True,sample_rule='first two of frozen ten per context under original M4 SHA order',
        sample_roster=[{k:r[k] for k in ('sample_id','context_index','query_id')} for r in coordinator],
        sample_count=10,contexts=5,independent_rater_packets=2,output_sha256=outputs,
        analyzer_sha256=sha(__file__),protocol_sha256=sha(HERE.parent/'E02_PREPARATION.md'),
        guide_sha256=sha(HERE/'RATER_GUIDE.md'),elapsed_seconds=elapsed,max_rss_kib=rss,
        paper_role='reference_only/reproducibility',section_id='source_evidence',item_id='E02',
        claim_links=['C02','R03','R04','R07'],next_route='Stop; await personnel, training/budget and threshold approvals.'))
    total=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    require(total<=100*1024**2,'storage budget exceeded')
    atomic(output/'status.json',dict(status='PREPARATION_COMPLETE',annotation_status='NOT_STARTED',
                                   human_annotations=0,output_bytes=total))
    print(json.dumps(dict(status='PREPARATION_COMPLETE',human_annotations=0,samples=10,
                         elapsed_seconds=elapsed,output_bytes=total)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();output=args.output.resolve();existed=output.exists()
    try:run(output)
    except Exception as exc:
        if not existed and output.is_dir():atomic(output/'status.json',dict(status='FAILED',error=f'{type(exc).__name__}: {exc}'))
        raise
