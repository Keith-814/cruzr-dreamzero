"""Generate a separately patched DreamZero training workspace for Cruzr.

The vendored snapshot is never edited. This is a reproducible candidate LoRA
configuration; GPU memory/weight compatibility still require server validation.
"""
import json
from pathlib import Path
import shutil
from .deployment import digest


def patch_once(path, old, new, count=1):
    text = path.read_text()
    if text.count(old) != count:
        raise ValueError(f'Upstream patch anchor changed: {path.name}')
    path.write_text(text.replace(old, new))


def training_config(fps):
    target = 'groot.vla.data.'
    modality = {key: {'_target_': target+'dataset.ModalityConfig', 'delta_indices': indices, 'modality_keys': keys}
                for key, indices, keys in (
                    ('video', list(range(25)), ['video.head_left', 'video.wrist_left']),
                    ('state', [0], ['state.joint_pos', 'state.gripper_pos']),
                    ('action', list(range(24)), ['action.joint_pos', 'action.gripper_pos']),
                    ('language', [0], ['annotation.task']))}
    modality['video']['eval_delta_indices'] = [0]
    transforms = [{'_target_': target+'transform.'+name, 'apply_to': '${modality_configs.cruzr_s2.video.modality_keys}', **options}
                  for name, options in (
                      ('VideoToTensor', {}), ('VideoCrop', {'scale': .95, 'mode': 'random'}),
                      ('VideoResize', {'height': '${image_resolution_height}', 'width': '${image_resolution_width}', 'interpolation': 'linear'}),
                      ('VideoToNumpy', {}))]
    for name in ('state', 'action'):
        keys = modality[name]['modality_keys']
        transforms.extend([
            {'_target_': target+'transform.StateActionToTensor', 'apply_to': keys},
            {'_target_': target+'transform.StateActionTransform', 'apply_to': keys,
             'normalization_modes': {key: 'q99' for key in keys}}])
    transforms.extend([
        {'_target_': target+'transform.ConcatTransform',
         'video_concat_order': modality['video']['modality_keys'],
         'state_concat_order': modality['state']['modality_keys'], 'action_concat_order': modality['action']['modality_keys']},
        '${model_specific_transform}'])
    return {
        'defaults': ['dreamzero/base_48_wan_fine_aug_relative', '_self_'],
        'cruzr_data_root': '???', 'max_state_dim': 64, 'use_global_metadata': False,
        'relative_action': True, 'relative_action_per_horizon': False,
        'relative_action_keys': ['joint_pos', 'gripper_pos'], 'max_chunk_size': 4,
        'modality_configs': {'cruzr_s2': modality},
        'transforms': {'cruzr_s2': {'_target_': target+'transform.ComposedModalityTransform', 'transforms': transforms}},
        'metadata_versions': {'cruzr_s2': '0221'}, 'fps': {'cruzr_s2': fps},
        'train_dataset': {
            '_target_': target+'dataset.lerobot_sharded.ShardedLeRobotMixtureDataset.from_mixture_spec',
            '_convert_': 'object', 'mixture_spec': [{'dataset_path': {'cruzr_s2': ['${cruzr_data_root}']},
                                                  'dataset_weight': 1., 'distribute_weights': True}],
            'dataset_class': target+'dataset.lerobot_sharded.ShardedLeRobotSubLangSingleActionChunkDatasetDROID',
            'all_modality_configs': '${modality_configs}', 'all_transforms': '${transforms}',
            'metadata_versions': '${metadata_versions}', 'fps': '${fps}',
            'dataset_kwargs': {'video_backend': 'decord', 'use_global_metadata': False, 'max_chunk_size': '${max_chunk_size}',
                               'relative_action': True, 'relative_action_keys': '${relative_action_keys}', 'relative_action_per_horizon': False},
            'mixture_kwargs': {'training': True, 'balance_dataset_weights': False, 'seed': 42, 'shard_sampling_rate': 1.}}}


def prepare_source(source, destination, fps):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns('__pycache__', '.venv*', '*.egg-info'))
    schema = destination/'groot/vla/data/schema/embodiment_tags.py'
    patch_once(schema, 'class EmbodimentTag(Enum):', 'class EmbodimentTag(Enum):\n    CRUZR_S2 = "cruzr_s2"')
    converter = destination/'scripts/data/convert_lerobot_to_gear.py'
    patch_once(converter, 'VALID_EMBODIMENT_TAGS = [', 'VALID_EMBODIMENT_TAGS = [\n    "cruzr_s2",')
    mapping = destination/'groot/vla/configs/model/dreamzero/transform/base.yaml'
    mapping.write_text(mapping.read_text()+'\n  cruzr_s2: 33\n')
    collator = destination/'groot/vla/model/dreamzero/transform/dreamzero_cotrain.py'
    for variable in ('processed_item', 'item'):
        anchor = '                    if num_views > 1 and elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.AGIBOT.value]:'
        # Each branch is distinguishable by the immediately following variable.
        old = anchor+'\n                        '+variable+' = '
        new = ('                    if elem["embodiment_id"] == embodiment_tag_mapping.get("cruzr_s2"):\n'
               f'                        {variable} = "A two-camera robot video: head in the top-left, physical right wrist in the bottom-left; right half is inactive. " + str({variable})\n'
               '                    elif num_views > 1 and elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.AGIBOT.value]:\n'
               f'                        {variable} = ')
        patch_once(collator, old, new)
    config = destination/'groot/vla/configs/data/dreamzero/cruzr_s2_relative.yaml'
    config.write_text('# @package _global_\n'+json.dumps(training_config(fps), indent=2)+'\n')
    paths = [schema, converter, mapping, collator, config]
    receipt = {'upstream_source': str(source.resolve()), 'embodiment': 'cruzr_s2', 'projector_id': 33,
               'camera_order': ['head_left', 'wrist_left'], 'layout': 'head top-left; right wrist bottom-left; right half black',
               'candidate_not_gpu_validated': True,
               'patched_files': {str(p.relative_to(destination)): digest(p) for p in paths}}
    (destination/'cruzr-overlay.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt
