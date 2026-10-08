"""Read-only validation; prints an evidence receipt, never runs models or writes files."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
CAMPAIGN = ROOT / 'outputs/mab/review_m3_formal/20261002-m3-six-policy-v3'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(path.read_text())


def validate():
    evidence = {}

    def remember(path):
        evidence[str(path.relative_to(ROOT))] = sha(path)

    pdf = ROOT / 'review/nference_Time_Latent_Me.pdf'
    assert sha(pdf) == '6b04a3c49b9bb68cccf282059be34f7dadc516e4069e42b204fcae99f68ffed3'
    remember(pdf)
    remember(ROOT / 'review/review')
    decision = load(HERE / 'decision.json')
    expected_review = [f'R{i:02d}' for i in range(1, 17)]
    expected_claim = [f'C{i:02d}' for i in range(1, 13)]
    assert decision['review_ids'] == expected_review
    assert decision['claim_ids'] == expected_claim
    assert decision['action'] == 'request_user_decision'
    assert decision['selected_route'] is None and not decision['submission_ready']
    assert not decision['paper_edited'] and decision['new_model_runs'] == 0
    for filename, ids in [('REVISION_LEDGER.md', expected_review), ('CLAIM_EVIDENCE.md', expected_claim)]:
        rows = re.findall(r'^\|([RC]\d\d)\|', (HERE / filename).read_text(), re.M)
        assert rows == ids, (filename, rows)
    link_count = 0
    for name in ['CLAIM_EVIDENCE.md', 'REVISION_LEDGER.md', 'M5_REPORT.md']:
        path = HERE / name
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            linked = (path.parent / target).resolve()
            assert linked.is_file(), (name, target)
            link_count += 1
        remember(path)
    remember(HERE / 'decision.json')
    remember(Path(__file__).resolve())
    for report in ['M0_REPORT.md', 'm1/RESULTS.md', 'm2/M2_REPORT.md', 'm3/M3_REPORT.md',
                   'm4/M4_A_REPORT.md', 'm4/M4_B1_REPORT.md', 'm4/M4_B2_REPORT.md']:
        remember(BASE / report)
    checked_core = 0
    for receipt_name in ['acceptance.json', 'acceptance_b1.json', 'acceptance_b2.json']:
        path = BASE / 'm4' / receipt_name
        receipt = load(path)
        assert receipt['status'] == 'PASS'
        runs = receipt.get('runs', receipt.get('compared_runs'))
        assert len(runs) == 2
        remember(path)
        for artifact, expected in receipt['identical_artifact_sha256'].items():
            for run in runs:
                artifact_path = ROOT / run / artifact
                assert sha(artifact_path) == expected, str(artifact_path)
                remember(artifact_path)
            checked_core += 1
    summary = load(CAMPAIGN / 'results.json')['summary']
    b2 = BASE / 'm4/run-20261008-b2'
    policies = load(b2 / 'policy_summary.json')
    expected_strict = dict(full=.202, random=.0392, last_written=.018,
                           cosine_only=.168, recency_only=.202, native_order=.118)
    assert set(summary) == set(expected_strict)
    for policy, expected in expected_strict.items():
        assert summary[policy]['question_count'] == 2500
        assert summary[policy]['strict_candidate_accuracy']['value'] == expected
        assert policies[policy]['strict'] == summary[policy]['strict_candidate_accuracy']
        assert policies[policy]['candidate_valid'] == summary[policy]['candidate_valid_rate']
    assert summary['full']['em']['value'] == .284
    comparisons = load(b2 / 'comparisons.json')
    for policy, values in dict(random=(407, 399, 8), last_written=(460, 470, -10),
                               cosine_only=(85, 90, -5), recency_only=(0, 0, 0),
                               native_order=(210, 205, 5)).items():
        row = comparisons[policy]['aggregate']
        assert tuple(row[k] for k in ['net_correct', 'invalid_correct_net', 'valid_wrong_correct_net']) == values
        assert values[0] == values[1] + values[2]
    remember(CAMPAIGN / 'results.json')
    return dict(status='PASS', stage='M5', review_items=16, claims=12,
                markdown_links_checked=link_count, m4_core_artifacts_checked=checked_core,
                m4_core_files_checked=checked_core * 2, m3_records_in_accepted_summary=15000,
                new_model_runs=0, validation_scope='documents, receipts, core hashes, accepted numeric summaries; not full parent re-audit',
                evidence_sha256=evidence)


if __name__ == '__main__':
    result = validate()
    prior = HERE / 'acceptance.json'
    if prior.exists():
        assert result == load(prior), 'Materials or evidence changed since receipt'
    print(json.dumps(result, indent=2, ensure_ascii=False))
