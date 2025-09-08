import json
import os

def process_conflict(data_path, output_path):

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for sample in data:
        conflict_tags = []

        explicit_conflicts = sample.get("explicit_conflict_tags", None)
        if explicit_conflicts:
            for tag in explicit_conflicts:
                select_item = next((item for item in sample["selector_output"] if item["tag"] == tag), None)
                exclude_item = next((item for item in sample["excluder_output"] if item["tag"] == tag), None)

                select_reason = select_item["selected_reason"] if select_item else None
                exclude_reason = exclude_item["excluded_reason"] if exclude_item else None

                conflict_tags.append({
                    "tag": tag,
                    "select_reason": select_reason,
                    "exclude_reason": exclude_reason
                })

        implicit_conflicts = sample.get("implicit_conflict_tags", None)
        if implicit_conflicts:
            for tag in implicit_conflicts:
                exclude_item = next((item for item in sample.get("selector_implict_reason", []) if item["tag"] == tag), None)
                select_item = next((item for item in sample.get("excluder_implict_reason", []) if item["tag"] == tag), None)

                exclude_reason = exclude_item["not_select_reason"] if exclude_item else None
                select_reason = select_item["not_exclude_reason"] if select_item else None

                conflict_tags.append({
                    "tag": tag,
                    "select_reason": select_reason,
                    "exclude_reason": exclude_reason
                })

        sample["conflict_tags"] = conflict_tags

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
