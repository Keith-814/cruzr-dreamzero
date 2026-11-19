"""Data contracts and generated training patches; no heavy training dependencies."""
import ast
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1]/'src'))
from cruzr_sim.dataset import check_contract, check_vectors, inspect_metadata
from cruzr_sim.finetune import prepare_source, training_config

ROOT = Path(__file__).parents[1]


class FinetuneContractTest(unittest.TestCase):
    def test_unverified_units_or_post_action_alignment_are_not_training_ready(self):
        contract = json.loads((ROOT/'configs/dataset.example.json').read_text())
        with self.assertRaises(ValueError):
            check_contract(contract)
        contract.update(semantics_verified=True, observation_precedes_action=True)
        check_contract(contract)
        contract['joint_unit'] = 'degree'
        with self.assertRaises(ValueError):
            check_contract(contract)

    def test_nonfinite_out_of_bounds_and_irregular_samples_are_rejected(self):
        state = np.zeros((25, 8))
        state[:, -1] = .04
        timestamps = np.arange(25)/15
        check_vectors(state, state.copy(), timestamps, 15)
        for index, value in ((0, 179), (1, np.nan), (7, .06)):
            action = state.copy()
            action[-1, index] = value
            with self.assertRaises(ValueError):
                check_vectors(state, action, timestamps, 15)
        with self.assertRaises(ValueError):
            check_vectors(state, state, timestamps*2, 15)

    def test_v3_is_rejected_instead_of_treating_shards_as_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'meta'
            path.mkdir()
            (path/'info.json').write_text('{"codebase_version":"v3.0"}')
            with self.assertRaisesRegex(ValueError, 'v3'):
                inspect_metadata(directory, {'fps': 15})

    def test_overlay_preserves_vendor_source_and_handles_two_view_language(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/'source', Path(directory)/'prepared'
            files = ('groot/vla/data/schema/embodiment_tags.py', 'scripts/data/convert_lerobot_to_gear.py',
                     'groot/vla/configs/model/dreamzero/transform/base.yaml',
                     'groot/vla/model/dreamzero/transform/dreamzero_cotrain.py')
            for name in files:
                target = source/name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT/'third_party/dreamzero'/name, target)
            (source/'groot/vla/configs/data/dreamzero').mkdir(parents=True)
            before = {name: (source/name).read_bytes() for name in files}
            receipt = prepare_source(source, output, 15)
            for name in files:
                self.assertEqual((source/name).read_bytes(), before[name])
            self.assertEqual(receipt['projector_id'], 33)
            text = (output/files[-1]).read_text()
            tree = ast.parse(text)
            collate = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'collate')
            names = ('AGIBOT', 'OXE_DROID', 'GR1_UNIFIED', 'MECKA_HANDS', 'XDOF', 'YAM')
            tags = types.SimpleNamespace(**{name: types.SimpleNamespace(value=name.lower()) for name in names})
            mapping = {name.lower(): index for index, name in enumerate(names)}
            mapping['cruzr_s2'] = 33
            scope = {'List': list, 'AutoTokenizer': object, 'ast': ast, 'np': np, 'EmbodimentTag': tags,
                     'torch': types.SimpleNamespace(from_numpy=lambda x: x)}
            exec(compile(ast.Module(body=[collate], type_ignores=[]), '<generated-collator>', 'exec'), scope)
            def tokenizer(values, **_):
                return values, [True]*len(values)
            for value in ('scan then return', "'scan then return'"):
                batch = scope['collate']([{'text': value, 'embodiment_id': 33}], tokenizer, 2, mapping)
                self.assertIn('physical right wrist', batch['text'][0])
            cfg = json.loads((output/'groot/vla/configs/data/dreamzero/cruzr_s2_relative.yaml').read_text().split('\n', 1)[1])
            self.assertEqual(cfg['fps']['cruzr_s2'], 15)
            self.assertEqual(cfg['modality_configs']['cruzr_s2']['video']['modality_keys'], ['video.head_left', 'video.wrist_left'])
