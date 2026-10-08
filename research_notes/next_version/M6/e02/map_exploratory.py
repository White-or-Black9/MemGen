"""Exploratory AI adjudication and source links; NOT human annotation acceptance."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
from itertools import combinations
import json
from pathlib import Path
import resource
import time

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / 'prepare-20261008-01'
ANNOTATIONS = ROOT / 'annotations_codex_single_rater.jsonl'
PLAN = ROOT.parent / 'E02_EXPLORATORY_PLAN.md'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dump(path, value, jsonl=False):
    if path.exists():
        raise ValueError(f'Will not overwrite {path}')
    if jsonl:
        text = ''.join(json.dumps(x, ensure_ascii=False, sort_keys=True) + '\n' for x in value)
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
    path.write_text(text)


def validate(annotations, questions, texts):
    if len(annotations) != len(questions):
        raise ValueError('Roster count mismatch')
    if len({a['sample_id'] for a in annotations}) != len(annotations):
        raise ValueError('Duplicate sample')
    if {a['sample_id'] for a in annotations} != set(questions):
        raise ValueError('Roster mismatch')
    spans = 0
    for a in annotations:
        q = questions[a['sample_id']]
        t = texts[q['context_index']]
        for key in ['context_sha256', 'question_sha256']:
            if a[key] != q[key]:
                raise ValueError(f'Identity mismatch: {key}')
        if hashlib.sha256(t.encode()).hexdigest() != a['context_sha256']:
            raise ValueError('Source hash mismatch')
        if a['status'] not in {'supported', 'ambiguous', 'not_locatable', 'conflict',
                               'insufficient_context', 'prior_only'}:
            raise ValueError('Invalid status')
        if any(type(i) is not int or not 0 <= i < len(q['candidates'])
               for i in a['chosen_candidate_indices']):
            raise ValueError('Candidate index invalid')
        set_ids = [s['set_id'] for s in a['evidence_sets']]
        if len(set(set_ids)) != len(set_ids):
            raise ValueError('Duplicate evidence set')
        for s in a['evidence_sets']:
            if s['sufficiency'] not in {'sufficient', 'partial', 'unknown'}:
                raise ValueError('Invalid sufficiency')
            for sp in s['spans']:
                start, end = sp['start'], sp['end']
                if (type(start) is not int or type(end) is not int or
                        not 0 <= start < end <= len(t) or t[start:end] != sp['text']):
                    raise ValueError('Invalid Unicode span')
                spans += 1
        if a['status'] == 'supported' and (len(a['chosen_candidate_indices']) != 1 or
                not any(s['sufficiency'] == 'sufficient' and s['spans']
                        for s in a['evidence_sets'])):
            raise ValueError('Supported requires unique candidate and sufficient spans')
    return spans


def adjudicate(annotations, questions, texts):
    """Public-text-only decisions. Never consult gold/slot/retrieval effects here."""
    result = deepcopy(annotations)
    for a in result:
        a['schema'] = 'E02-exploratory-adjudication/v1'
        a['original_status'] = a['status']
        a['original_chosen_candidate_indices'] = list(a['chosen_candidate_indices'])
        a['original_rater_id'] = a.pop('rater_id')
        a['adjudicator_type'] = 'AI_assistant_not_independent_human'
        a['user_review'] = 'User verbally confirmed review; no item-level human review record supplied'
        a['original_elapsed_minutes_unverified'] = a.pop('elapsed_minutes')
        a['original_completed_at'] = a.pop('completed_at')
        a['decision_basis'] = 'Existing source spans provisionally retained; no independent human verification'
        if a['sample_id'] == 'E02P-c2-01':
            a['status'] = 'ambiguous'
            a['chosen_candidate_indices'] = [4, 5]
            t = texts[questions[a['sample_id']]['context_index']]
            quote = ('Well, repeat it, -- repeat it, I beg of you, that I may at last believe it! '
                     'Tell me for the hundredth time that you refuse my love, which had your '
                     "mother's sanction. Make me understand once for all that you are trifling "
                     'with my happiness, that my life or death are nothing to you. Ah, to have '
                     'dreamed for ten years of being your husband, Abrielle, and to lose that '
                     'hope, which was the only stay of my existence!')
            start = t.index(quote)
            acknowledgement = 'Yes, that is very true, Abrielle'
            ack_start = t.index(acknowledgement, start)
            s = a['evidence_sets'][0]
            s['spans'] = [s['spans'][0],
                dict(start=start, end=start+len(quote), text=quote, role='target_event',
                     reason='Nearby reaction expresses lost hope, not a distant conversation with Blaire'),
                dict(start=ack_start, end=ack_start+len(acknowledgement), text=acknowledgement,
                     role='supporting', reason='Acknowledgement occurs later within the same conversation')]
            s['sufficiency'] = 'partial'
            s['required_facts'] = ['Repeated refusal precedes lost-hope reaction',
                                   'Acknowledgement occurs later in the conversation',
                                   'A unique full candidate paraphrase remains unresolved']
            s['rationale'] = ('Original distant despair quote is not a sufficient next-event link. '
                              'Nearby dialogue supports loss/despair but does not uniquely distinguish '
                              'the complete paraphrases 4 and 5. These are tentative, not accepted answers.')
            a['notes'] = 'Do not use this question for sufficient evidence mapping or answer coverage'
            a['decision_basis'] = s['rationale']
        elif a['sample_id'] == 'E02P-c1-01':
            a['decision_basis'] = ('Retain provisional temporal conflict: relatives remark precedes '
                                   'cheese explanation. Not a confirmed dataset label error.')
        elif a['sample_id'] == 'E02P-c3-02':
            a['status'] = 'ambiguous'
            a['chosen_candidate_indices'] = [0]
            a['notes'] = ('Candidate 0 is only a tentative action match. Mother versus narrator '
                          'and Mrs. Kayla/bare Kayla aliasing is unresolved; not a unique supported answer.')
            a['decision_basis'] = ('Fainting subject is the mother. Question already uses bare Kayla '
                                   'in the door-opening clue, so alias ambiguity precludes claiming '
                                   'a confirmed subject error or a fully supported answer.')
        if a['status'] in {'conflict', 'ambiguous'}:
            for s in a['evidence_sets']:
                s['sufficiency'] = 'partial'
                s['answer_support'] = False
        a['source_mapping_eligible'] = (a['status'] == 'supported' and
            any(s['sufficiency'] == 'sufficient' for s in a['evidence_sets']))
    return result


def span_chunks(span, chunks, text):
    """Exact non-whitespace source-character coverage, excluding joined-sentence gaps."""
    required = {i for i in range(span['start'], span['end']) if not text[i].isspace()}
    hits = {}
    for c in chunks:
        covered = set()
        for start, end in c['sentence_spans']:
            covered.update(i for i in range(max(start, span['start']), min(end, span['end']))
                           if i in required)
        if covered:
            hits[c['chunk_index']] = len(covered)
    # Chunks are non-overlapping here, but compute union explicitly for correctness.
    all_covered = {i for c in chunks for start, end in c['sentence_spans']
                   for i in range(max(start, span['start']), min(end, span['end'])) if i in required}
    return dict(chunk_indices=sorted(hits), nonwhitespace_chars_by_chunk=hits,
                required_nonwhitespace_chars=len(required),
                covered_nonwhitespace_chars=len(all_covered),
                fully_covered=bool(required) and all_covered == required)


def link(required, slot):
    direct = {slot['direct_source_chunk']}
    indirect = set(slot['indirect_possible_source_chunks'])
    return dict(slot_index=slot['slot_index'], slot_version=slot['slot_version'],
                direct_source_chunk=slot['direct_source_chunk'],
                direct_intersection=sorted(required & direct),
                indirect_possible_intersection=sorted(required & indirect),
                source_link=('direct_source_candidate' if required & direct else
                             'indirect_possible_only' if required & indirect else
                             'no_recorded_source_link_NOT_semantic_negative'))


def map_sources(adjudicated, roster, chunks, slots, texts):
    mapped = []
    for a in adjudicated:
        r = roster[a['sample_id']]
        c = r['context_index']
        for evidence in a['evidence_sets']:
            mappings = [span_chunks(sp, [x for x in chunks if x['context_index'] == c], texts[c])
                        for sp in evidence['spans']]
            required = set().union(*(set(m['chunk_indices']) for m in mappings))
            eligible = (a['source_mapping_eligible'] and evidence['sufficiency'] == 'sufficient'
                        and bool(mappings) and all(m['fully_covered'] for m in mappings))
            for seed in sorted({s['seed'] for s in slots if s['context_index'] == c}):
                bank = [s for s in slots if s['context_index'] == c and s['seed'] == seed]
                direct_pairs = []
                potential_pairs = []
                if eligible:
                    for x, y in combinations(bank, 2):
                        pair = [x['slot_index'], y['slot_index']]
                        if required <= {x['direct_source_chunk'], y['direct_source_chunk']}:
                            direct_pairs.append(pair)
                        if required <= set(x['potential_source_chunks']) | set(y['potential_source_chunks']):
                            potential_pairs.append(pair)
                mapped.append(dict(sample_id=a['sample_id'], context_index=c, query_id=r['query_id'],
                    seed=seed, set_id=evidence['set_id'], status=a['status'], eligible=eligible,
                    required_source_chunks=sorted(required), span_mapping=mappings,
                    slots=[link(required, s) for s in bank],
                    two_slot_direct_all_source_pairs=direct_pairs,
                    two_slot_potential_all_source_pairs_NOT_retention=potential_pairs,
                    latent_content_contribution='unknown', downstream_benefit='not_tested'))
    return mapped


def run(output):
    if output.exists():
        raise ValueError('Output already exists')
    started = time.monotonic()
    files = [p for p in PACKAGE.rglob('*') if p.is_file()] + [ANNOTATIONS, ROOT/'ANNOTATION_NOTE.md', PLAN, Path(__file__)]
    before = {str(p): sha(p) for p in files}
    manifest = json.loads((PACKAGE/'manifest.json').read_text())
    for name, digest in manifest['output_sha256'].items():
        if sha(PACKAGE/name) != digest:
            raise ValueError(f'Frozen package changed: {name}')
    questions = {q['sample_id']: q for q in rows(PACKAGE/'rater_A/questions.jsonl')}
    texts = {c: (PACKAGE/f'rater_A/contexts/c{c}.txt').read_text() for c in range(5)}
    annotations = rows(ANNOTATIONS)
    original_spans = validate(annotations, questions, texts)
    adjudicated = adjudicate(annotations, questions, texts)
    adjudicated_spans = validate(adjudicated, questions, texts)
    output.mkdir()
    dump(output/'adjudicated_ai_exploratory.jsonl', adjudicated, True)
    # Freeze AI decisions before opening coordinator fields, including gold-bearing roster.
    adjudication_sha = sha(output/'adjudicated_ai_exploratory.jsonl')
    roster = {r['sample_id']: r for r in rows(PACKAGE/'coordinator_only/roster_and_gold.jsonl')}
    chunks = rows(PACKAGE/'coordinator_only/chunk_offsets.jsonl')
    slots = rows(PACKAGE/'coordinator_only/slot_sources.jsonl')
    for r in roster.values():
        if r['question_sha256'] != questions[r['sample_id']]['question_sha256']:
            raise ValueError('Coordinator question identity mismatch')
    for c in chunks:
        reconstructed = ' '.join(texts[c['context_index']][a:b] for a,b in c['sentence_spans'])
        if hashlib.sha256(reconstructed.encode()).hexdigest() != c['chunk_sha256']:
            raise ValueError('Chunk reconstruction mismatch')
    chunk_lookup = {(c['context_index'], c['chunk_index']):c for c in chunks}
    if len({(s['seed'],s['context_index'],s['slot_index']) for s in slots}) != len(slots):
        raise ValueError('Duplicate slot identity')
    for s in slots:
        if s['direct_source_chunk_sha256'] != chunk_lookup[(s['context_index'],s['direct_source_chunk'])]['chunk_sha256']:
            raise ValueError('Slot direct source hash mismatch')
        if set(s['potential_source_chunks']) != ({s['direct_source_chunk']} | set(s['indirect_possible_source_chunks'])):
            raise ValueError('Potential source chain inconsistent')
    mapped = map_sources(adjudicated, roster, chunks, slots, texts)
    dump(output/'source_slot_mapping.jsonl', mapped, True)
    by_sample = []
    for a in adjudicated:
        ms = [m for m in mapped if m['sample_id']==a['sample_id']]
        by_sample.append(dict(sample_id=a['sample_id'], status=a['status'],
            eligible=a['source_mapping_eligible'],
            required_source_chunks=ms[0]['required_source_chunks'],
            seeds_with_direct_pair=sorted({m['seed'] for m in ms if m['two_slot_direct_all_source_pairs']}),
            seeds_with_potential_pair=sorted({m['seed'] for m in ms if m['two_slot_potential_all_source_pairs_NOT_retention']}),
            direct_pair_counts=[len(m['two_slot_direct_all_source_pairs']) for m in ms]))
    summary = dict(status='EXPLORATORY_SOURCE_MAPPING_ONLY', sample_count=len(adjudicated),
        status_counts=dict(Counter(a['status'] for a in adjudicated)), original_validated_spans=original_spans,
        adjudicated_validated_spans=adjudicated_spans, source_chunk_count=len(chunks), current_slot_count=len(slots),
        eligible_question_count=sum(a['source_mapping_eligible'] for a in adjudicated),
        direct_pair_question_count=sum(bool(x['seeds_with_direct_pair']) for x in by_sample),
        potential_pair_question_count=sum(bool(x['seeds_with_potential_pair']) for x in by_sample),
        denominator_all_questions=len(adjudicated), per_sample=by_sample,
        independent_context_upper_bound=5, human_independent_rater_count_verified=0,
        user_verbal_review=True, inter_rater_agreement=None, formal_E02_gate='NOT_PASSED',
        semantic_negative_slots_defined=False, latent_retention='UNKNOWN', new_gpu_runs=0,
        adjudication_frozen_sha256=adjudication_sha)
    dump(output/'summary.json',summary)
    after = {str(p):sha(p) for p in files}
    if before != after or sha(output/'adjudicated_ai_exploratory.jsonl') != adjudication_sha:
        raise ValueError('Input or frozen adjudication changed')
    core = {name:sha(output/name) for name in ['adjudicated_ai_exploratory.jsonl','source_slot_mapping.jsonl','summary.json']}
    dump(output/'acceptance.json',dict(status='PASS_TECHNICAL_EXPLORATORY_ONLY',
        formal_human_acceptance=False, input_preserved=True, input_sha256=before, core_sha256=core,
        analyzer_sha256=sha(Path(__file__)), elapsed_seconds=time.monotonic()-started,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        next_route='STOP: request separate E03 or formal independent human annotation approval'))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
