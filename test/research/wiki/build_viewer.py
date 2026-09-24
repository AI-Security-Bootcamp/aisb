"""Build a standalone viewer for the single classification pass.

Reads: simple_outputs/02-records.jsonl and 01-codebook.json.
Writes: classification.html. No model calls or new summaries.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INPUTS = HERE / 'simple_outputs'
OUTPUT = HERE / 'classification.html'

records = [json.loads(line) for line in (INPUTS / '02-records.jsonl').read_text().splitlines()]
source_ids = [json.loads(line)['id'] for line in (INPUTS / '00-wiki.jsonl').read_text().splitlines()]
assert [r['id'] for r in records] == source_ids, 'Classification output is incomplete or out of order'
codebook = json.loads((INPUTS / '01-codebook.json').read_text())
# The generic viewer also supports older hierarchical investigations. All labels
# in this pipeline are top-level rows; no parents are added to the codebook.
labels = [dict(label, parent_id=None) for label in codebook['labels']] + [
    {'id': 'other', 'parent_id': None, 'name': 'Other'},
    {'id': 'insufficient_context', 'parent_id': None, 'name': 'Insufficient context'},
]
label_names = {label['id']: label['name'] for label in labels}
assert len({r['id'] for r in records}) == len(records), 'Duplicate source IDs'

# Consecutive observations with the same label form a display episode. Split at
# five-minute gaps; the resulting bars do not claim continuous task execution.
groups = []
previous_label = None
previous_time = None
for record in records:
    assert record['label'] in label_names, record['id']
    assert record['quote'] in record['text'], record['id']
    stamp = datetime.fromisoformat(record['time']) if record['time'] else None
    if stamp is not None and stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
        # Preserve the source string and disclose the plotting assumption.
        record['plot_time'] = record['time'] + 'Z'
        record['time_grade'] = 'Minute precision; timezone omitted in source, plotted as UTC following surrounding records'
    close = stamp is not None and previous_time is not None and 0 <= (stamp - previous_time).total_seconds() <= 300
    if record['label'] != previous_label or not close:
        groups.append([])
    groups[-1].append(record)
    previous_label, previous_time = record['label'], stamp

stories = []
for number, group in enumerate(groups):
    highlights = []
    for record in group:
        if record['quote']:
            start = record['text'].index(record['quote'])
            highlights.append({
                'id': record['id'], 'start': start,
                'end': start + len(record['quote']), 'quote': record['quote'],
            })
    label = group[0]['label']
    span = group[0]['id'] if len(group) == 1 else f"{group[0]['id']}–{group[-1]['id']}"
    stories.append({
        'id': f'episode-{number}', 'title': f'{label_names[label]} · {span}',
        'summary': f'{len(group)} consecutive records assigned this label. Outcomes are not assessed by the activity classifier.',
        'labels': [label], 'source_ids': [r['id'] for r in group],
        'highlights': highlights,
    })

data = {
    'title': 'Wiki: classified activities',
    'description': 'DeepSeek V4.1 Flash summaries → flat codebook → one classification pass with surrounding context. Posts are external communications; reported success is not independently verified. Click a bar for source text, or a label to filter that activity.',
    'track': 'wiki', 'flat_labels': True,
    'units': records, 'labels': labels, 'stories': stories,
}
# Keep source HTML inert, including any closing script tags in the wiki posts.
encoded = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
OUTPUT.write_text((ROOT / 'viewer.html').read_text().replace(
    '<script id="embedded-data" type="application/json">null</script>',
    f'<script id="embedded-data" type="application/json">{encoded}</script>',
))
print(f'Wrote {OUTPUT}: {len(records)} records, {len(codebook["labels"])} flat labels.')
