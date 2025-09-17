import json
from typing import List, Dict, Any


def process_samples(data_path: str) -> List[Dict[str, Any]]:

    # Load data
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    updated = []

    for sample in data:
        # Build helper maps
        caption_map = {e['tag']: e for e in sample.get('image_caption', [])}
        selector_map = {e['tag']: e['selected_reason'] for e in sample.get('selector_output', [])}
        key_list: List[Dict[str, Any]] = []

        judge_output = sample.get('judge_output', None)

        if judge_output is None:
            for tag, reason in selector_map.items():
                caption_entry = caption_map.get(tag, {})
                key_list.append({
                    'tag': tag,
                    'attributes': caption_entry.get('attributes'),
                    'caption': caption_entry.get('caption'),
                    'selected_reason': reason
                })
        else:
            judge_map = {e['tag']: e for e in sample.get('judge_output', [])}
            conflict = sample.get('conflict_tags') or []
            conflict_tags = {c['tag']: c for c in conflict}
            # Include non-conflict selector tags
            for tag, reason in selector_map.items():
                if tag not in conflict_tags:
                    caption_entry = caption_map.get(tag, {})
                    key_list.append({
                        'tag': tag,
                        'attributes': caption_entry.get('attributes'),
                        'caption': caption_entry.get('caption'),
                        'selected_reason': reason
                    })
            # Evaluate each conflict tag
            for tag, conf in conflict_tags.items():
                judge = judge_map.get(tag)
                if not judge:
                    continue
                sel_score = judge.get('select_score', {}).get('score', 0)
                excl_score = judge.get('exclude_score', {}).get('score', 0)
                if sel_score >= excl_score:
                    caption_entry = caption_map.get(tag, {})
                    key_list.append({
                        'tag': tag,
                        'attributes': caption_entry.get('attributes'),
                        'caption': caption_entry.get('caption'),
                        'selected_reason': conf.get('select_reason')
                    })

        sample['key_imformation'] = key_list
        updated.append(sample)

    return updated

