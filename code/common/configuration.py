"""Shared geometry values and references used by the RIDGE protocol templates."""
from pathlib import Path
import json

FIELDS = ('elongation_factor', 'whiteness_threshold', 'angle_threshold',
          'compactness_threshold', 'length_diff_threshold', 'rects_min_area_factor',
          'rects_max_area_factor', 'nms_threshold')


def load_shared_geometry(root):
    values = json.loads((Path(root) / 'parameters/shared_geometry.json').read_text(encoding='utf-8'))
    for dataset, geometry in values.items():
        if set(geometry) != set(FIELDS):
            raise ValueError(f'Expected eight shared geometry settings: {dataset}')
    return values


def resolve_shared(value, geometry):
    """Resolve {"$shared": "field"} entries without modifying the template."""
    if isinstance(value, list):
        return [resolve_shared(item, geometry) for item in value]
    if isinstance(value, dict):
        if set(value) == {'$shared'}:
            return geometry[value['$shared']]
        return {key: resolve_shared(item, geometry) for key, item in value.items()}
    return value
