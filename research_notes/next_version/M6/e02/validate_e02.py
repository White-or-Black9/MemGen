"""Read-only acceptance of preparation, never evaluates nonexistent human annotations."""
import collections
import json
from pathlib import Path
from prepare_e02 import HERE, PARENT, M4, blank_form, candidates, check_blind, check_blank, lines, read, require, select_pilot, sha, text_sha, verify_sources


def validate():
    verify_sources()
    runs=[HERE/n for n in ('prepare-20261008-01','prepare-20261008-02')]
    manifests=[read(p/'manifest.json') for p in runs]
    expected=select_pilot(lines(M4/'run-20261008-a1/blind_sample.jsonl'))
    for p,m in zip(runs,manifests):
        require(m['status']=='PREPARATION_COMPLETE' and m['annotation_status']=='NOT_STARTED','wrong state')
        require(read(p/'status.json')['human_annotations']==m['human_annotations']==0,'human labels fabricated')
        require(m['sample_count']==10 and m['contexts']==5 and m['independent_rater_packets']==2,'coverage')
        require(m['new_model_runs']==0 and m['gpu_used'] is False and m['parent_preserved'],'scope violation')
        require(m['analyzer_sha256']==sha(HERE/'prepare_e02.py'),'preparer changed')
        require(m['protocol_sha256']==sha(HERE.parent/'E02_PREPARATION.md') and m['guide_sha256']==sha(HERE/'RATER_GUIDE.md'),'protocol changed')
        require(m['elapsed_seconds']<600 and m['max_rss_kib']<=4*1024**2,'resource cap')
        require(sum(x.stat().st_size for x in p.rglob('*') if x.is_file())<=100*1024**2,'storage cap')
        for n,h in m['output_sha256'].items():require(sha(p/n)==h,'package drift: '+n)
        for role in ('rater_A','rater_B'):
            packet=p/role
            require({str(f.relative_to(packet)) for f in packet.rglob('*') if f.is_file()}==
                    {'README.md','questions.jsonl','annotations_blank.jsonl',*(f'contexts/c{i}.txt' for i in range(5))},'unexpected initial file')
            questions=lines(packet/'questions.jsonl');forms=lines(packet/'annotations_blank.jsonl')
            require(len(questions)==len(forms)==10,'packet denominator')
            require(collections.Counter(q['context_index'] for q in questions)=={i:2 for i in range(5)},'context balance')
            for q,f,old in zip(questions,forms,expected):
                check_blind(q);check_blind(f);check_blank(f)
                require(f==blank_form(q),'blank form changed')
                require(q['question']==old['question'] and text_sha(q['question'])==q['question_sha256'],'question changed')
                require(q['candidates']==candidates(old['question']),'candidates changed')
                require(sha(packet/q['context_file'])==q['context_sha256'],'context bytes changed')
            for f in packet.rglob('*'):
                if f.is_file():require(sha(f)==sha(p/('rater_B' if role=='rater_A' else 'rater_A')/f.relative_to(packet)),'asymmetric packets')
        gate=read(p/'coordinator_only/release_gate.json')
        require(gate['status']=='NOT_RELEASED' and gate['annotation_status']=='NOT_STARTED','release not approved')
        require(all(gate[k] is None for k in ('rater_A','rater_B','adjudicator','approval_record')),'personnel/approval invented')
        require(all(v is None for v in gate['approved_thresholds'].values()),'threshold invented')
        adjudication=lines(p/'coordinator_only/adjudication_blank.jsonl')
        require(len(adjudication)==10 and all(r['adjudicated_status'] is None and not r['accepted_evidence_sets'] for r in adjudication),'adjudication fabricated')
    require(manifests[0]['output_sha256']==manifests[1]['output_sha256'],'rerun differs')
    result=dict(status='PASS_PREPARATION_ONLY',experiment_id='M6-E02-PREP',compared_runs=[p.name for p in runs],
        identical_artifact_sha256=manifests[0]['output_sha256'],frozen_questions=10,contexts=5,
        initial_rater_packets=2,blank_phase1_forms=20,human_annotations=0,
        annotation_status='NOT_STARTED',release_status='NOT_RELEASED',new_tests_passed=13,
        semantic_mapping_verified=False,latent_retention_verified=False,new_model_runs=0,gpu_used=False)
    if (HERE/'acceptance.json').exists():
        require(read(HERE/'acceptance.json')['identical_artifact_sha256']==result['identical_artifact_sha256'],'receipt changed')
    return result


if __name__=='__main__':print(json.dumps(validate(),indent=2))
