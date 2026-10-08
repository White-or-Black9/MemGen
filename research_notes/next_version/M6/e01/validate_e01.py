"""Read-only validation of E01 receipts and frozen JSON evidence."""
import json
from pathlib import Path
from audit_e01 import CONTRACT, HERE, ROOT, read, require, sha


def validate():
    runs = [HERE / name for name in ('run-20261008-01', 'run-20261008-02')]
    manifests = [read(p / 'manifest.json') for p in runs]
    for p, m in zip(runs, manifests):
        require(read(p/'status.json')['status'] == 'COMPLETE', 'incomplete run')
        require(m['parent_contract_sha256'] == CONTRACT, 'contract drift')
        require(m['analyzer_sha256'] == sha(HERE/'audit_e01.py'), 'analyzer drift')
        require(m['execution_contract_sha256'] == sha(HERE.parent/'E01_EXECUTION.md'), 'execution contract drift')
        require(m['parent_inventory_unchanged'] and m['m4_inventory_unchanged'], 'parent change')
        require(not any(m[k] for k in ('torch_imported','transformers_imported','gpu_used','model_runs')), 'model/GPU used')
        for name, h in m['core_artifact_sha256'].items():
            require(sha(p/name) == h, 'core artifact drift: '+name)
        require(m['max_rss_kib'] <= 4*1024**2 and m['elapsed_seconds'] < 1800, 'resource cap')
    require(manifests[0]['core_artifact_sha256'] == manifests[1]['core_artifact_sha256'], 'repeat differs')
    receipt = read(HERE/'acceptance.json')
    require(receipt['status'] == 'PASS' and receipt['identical_artifact_sha256'] == manifests[0]['core_artifact_sha256'],
            'acceptance receipt differs')
    require(manifests[0]['parent_json_sha256'] == manifests[1]['parent_json_sha256'], 'parent indices differ')
    parent = ROOT/'outputs/mab/review_m3_formal/20261002-m3-six-policy-v3'
    for name, h in manifests[0]['parent_json_sha256'].items():
        require(sha(parent/name) == h, 'live parent drift: '+name)
    summary = read(runs[0]/'summary.json')
    require((summary['groups'], summary['jobs'], summary['unique_questions'], summary['policy_records_verified'])
            == (2500,25,500,15000), 'coverage mismatch')
    return dict(status='PASS', exp_id='M6-E01', compared_runs=[str(p.relative_to(ROOT)) for p in runs],
                identical_artifact_sha256=manifests[0]['core_artifact_sha256'],
                parent_json_hashes_verified=len(manifests[0]['parent_json_sha256']),
                model_runs=0,gpu_used=False,
                scope='Coverage, checksums, paired records, deterministic rerun and frozen-parent preservation; not semantic validity.')


if __name__ == '__main__':
    print(json.dumps(validate(), indent=2))
