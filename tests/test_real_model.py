import unittest

from aiqiang import features
from aiqiang.markers import markers
from common import ROOT, assert_nested, read_json

REAL_MODEL = ROOT / 'aiqiang' / 'model.json'
REAL_GOLDEN = ROOT / 'tests' / 'golden_real.json'


@unittest.skipUnless(REAL_MODEL.is_file() and REAL_GOLDEN.is_file(), 'Real model and golden not both present')
class RealModelTests(unittest.TestCase):
    def test_real_reference_values(self):
        model = read_json(REAL_MODEL)
        for case in read_json(REAL_GOLDEN)['cases']:
            with self.subTest(name=case.get('name', 'real')):
                text = case['text']
                self.assertEqual(features.normalize(text), case['normalized'])
                self.assertEqual(features.distort(features.normalize(text)), case['distorted'])
                self.assertEqual({row['name']: row['count'] for row in markers(text)}, case['markers'])
                for key, value in features.predict(model, text).items():
                    assert_nested(self, value, case[key])
                if 'pieces' in case:
                    assert_nested(self, features.split_pieces(text, model.get('long_text')), case['pieces'])
                if 'doc' in case:
                    assert_nested(self, features.predict_doc(model, text), case['doc'])
                if 'features' in case:
                    for fs, expected in zip(model['feature_sets'], case['features']):
                        assert_nested(self, features.featurize(text, fs), expected)


if __name__ == '__main__':
    unittest.main()
