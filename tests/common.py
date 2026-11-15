"""Fixture loading and recursive float comparison for numerical parity."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'tests' / 'fixtures' / 'toy_model.json'
GOLDEN_PATH = MODEL_PATH.with_name('toy_golden.json')


def read_json(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def assert_nested(test, actual, expected):
    if isinstance(expected, dict):
        test.assertEqual(set(actual), set(expected))
        for key, value in expected.items():
            assert_nested(test, actual[key], value)
    elif isinstance(expected, list):
        test.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            assert_nested(test, a, b)
    elif isinstance(expected, float):
        test.assertAlmostEqual(actual, expected, delta=1e-9)
    else:
        test.assertEqual(actual, expected)
