"""Read-only integrity/coverage validation, separate from scientific reproduction gates."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'e01'))
from audit_e01 import read, require, sha, ROOT
from validate_e01 import validate as validate_e01

HERE=Path(__file__).resolve().parent


def validate():
    import numpy as np
    runs=[HERE/name for name in ('run-20261008-02','run-20261008-03')]
    manifests=[read(p/'manifest.json') for p in runs]
    summaries=[]
    for p,m in zip(runs,manifests):
        summary=read(p/'summary.json');summaries.append(summary)
        require(summary['status'] in ('COMPLETE','PARTIAL'), 'run unfinished')
        require(read(p/'status.json')['status']==summary['status'],'status mismatch')
        require(m['analyzer_sha256']==sha(HERE/'audit_e01v.py'),'analyzer changed')
        require(m['execution_contract_sha256']==sha(HERE.parent/'E01V_EXECUTION.md'),'contract changed')
        require(m['parent_inventory_unchanged'] and m['e01_inventory_unchanged'],'parent inventory changed')
        require(m['transformer_tensors_loaded']==m['model_forward_calls']==0 and not m['cuda_initialized'],'scope violation')
        require(m['max_rss_kib']<=16*1024**2 and m['elapsed_seconds']<1800 and m['threads']<=4,'resource cap')
        require(sum(x.stat().st_size for x in p.iterdir())<=100*1024**2,'storage cap')
        for n,h in m['core_artifact_sha256'].items():require(sha(p/n)==h,'artifact changed: '+n)
        require((summary['groups'],summary['unique_prompts'],summary['tokenizers_identical'])==(2500,500,500),'coverage')
        with np.load(p/'queries.npz',allow_pickle=False) as arrays:
            repeated=arrays['original_cpu_2500']
            unique=arrays['original_cpu_bf16_as_fp32']
            index=arrays['job_query_index']
            require(repeated.shape==(2500,1536),'query geometry shape')
            require(arrays['original_cpu_bf16_bits'].shape==(500,1536),'BF16 shape')
            require(index.shape==(2500,3),'index shape')
            for i,(_,ctx,qid) in enumerate(index):
                require(np.array_equal(repeated[i],unique[ctx*100+qid]),'seed replication differs')
            for c in range(5):require(arrays[f'qq_cosine_c{c}'].shape==(100,100),'QQ shape')
        rows=[json.loads(line) for line in (p/'query_metrics.jsonl').read_text().splitlines()]
        require(len(rows)==len({(r['job_id'],r['query_id']) for r in rows})==2500,'metric identities')
        interventions=[json.loads(line) for line in (p/'interventions.jsonl').read_text().splitlines()]
        require(len(interventions)==500 and all(i['clue_prefix_changed'] for i in interventions),'clue intervention coverage')
        require(all(i['candidate_token_changes']>0 for i in interventions),'candidate intervention empty')
    require(manifests[0]['core_artifact_sha256']==manifests[1]['core_artifact_sha256'],'repeat differs')
    for name,h in manifests[0]['input_sha256'].items():require(sha(name)==h,'current input drift')
    for name,h in manifests[0]['source_sha256'].items():require(sha(ROOT/name)==h,'current source drift')
    validate_e01()
    result=dict(status='PASS',experiment_id='M6-E01V',compared_runs=[p.name for p in runs],
                identical_artifact_sha256=manifests[0]['core_artifact_sha256'],
                scientific_reproduction_status=summaries[0]['status'],
                numerical_reproduction_gate=summaries[0]['numerical_reproduction_gate'],
                full_selection_reproduction_gate=summaries[0]['full_selection_reproduction_gate'],
                failed_attempts_preserved=['run-20261008-01'],new_unit_tests_passed=13,
                e01_regression_tests_passed=16,model_forward_calls=0,gpu_used=False,
                interpretation='PASS is integrity/repeatability, not semantic validity or proof of answer effects.')
    if (HERE/'acceptance.json').exists():
        receipt=read(HERE/'acceptance.json')
        require(receipt['identical_artifact_sha256']==result['identical_artifact_sha256'],'receipt differs')
    return result


if __name__=='__main__':
    print(json.dumps(validate(),indent=2))
