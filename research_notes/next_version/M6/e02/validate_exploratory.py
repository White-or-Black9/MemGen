"""Read-only exploratory replay. Never grants formal human acceptance."""
import json
from pathlib import Path
import map_exploratory as m


def main():
    roots=[m.ROOT/f'exploratory-20261008-{n}' for n in ['01','02']]
    receipts=[json.loads((r/'acceptance.json').read_text()) for r in roots]
    if receipts[0]['core_sha256']!=receipts[1]['core_sha256']:
        raise ValueError('Core differs between runs')
    for root,receipt in zip(roots,receipts):
        for path,digest in receipt['input_sha256'].items():
            if m.sha(Path(path))!=digest:raise ValueError(f'Input changed: {path}')
        for name,digest in receipt['core_sha256'].items():
            if m.sha(root/name)!=digest:raise ValueError(f'Core changed: {name}')
        qs={r['sample_id']:r for r in m.rows(m.PACKAGE/'rater_A/questions.jsonl')}
        texts={c:(m.PACKAGE/f'rater_A/contexts/c{c}.txt').read_text() for c in range(5)}
        expected=m.adjudicate(m.rows(m.ANNOTATIONS),qs,texts)
        actual=m.rows(root/'adjudicated_ai_exploratory.jsonl')
        if actual!=expected:raise ValueError('Adjudication replay differs')
        m.validate(actual,qs,texts)
        roster={r['sample_id']:r for r in m.rows(m.PACKAGE/'coordinator_only/roster_and_gold.jsonl')}
        mapping=m.map_sources(actual,roster,
            m.rows(m.PACKAGE/'coordinator_only/chunk_offsets.jsonl'),
            m.rows(m.PACKAGE/'coordinator_only/slot_sources.jsonl'),texts)
        if json.loads(json.dumps(mapping))!=m.rows(root/'source_slot_mapping.jsonl'):
            raise ValueError('Mapping replay differs')
        summary=json.loads((root/'summary.json').read_text())
        if summary['formal_E02_gate']!='NOT_PASSED' or summary['inter_rater_agreement'] is not None:
            raise ValueError('Invalid human acceptance claim')
    print(json.dumps(dict(status='PASS_TECHNICAL_EXPLORATORY_ONLY',runs=2,
        core_files_identical=3,adjudication_and_mapping_replayed=True,
        input_files_preserved=len(receipts[0]['input_sha256']),
        formal_human_acceptance=False,new_gpu_runs=0),indent=2))


if __name__=='__main__':main()
