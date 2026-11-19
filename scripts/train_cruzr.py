#!/usr/bin/env python3
"""Plan, compose or launch candidate DreamZero-Cruzr LoRA training.

Default is plan-only. Four-rank TRAINING differs from upstream one/two-rank
INFERENCE; memory and checkpoint compatibility remain server acceptance items.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cruzr_sim.deployment import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('prepared', 'checkpoint', 'wan', 'tokenizer', 'output-dir'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=20)
    parser.add_argument('--deepspeed', type=Path, help='Optional reviewed ZeRO/offload JSON; no automatic memory guarantee')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--compose', action='store_true', help='Resolve Hydra configuration without training')
    mode.add_argument('--launch', action='store_true', help='Actually launch training in the current model environment')
    args = parser.parse_args()
    prepared, output = args.prepared.resolve(), args.output_dir.resolve()
    report = json.loads((prepared/'prepare-report.json').read_text())
    if report.get('success') is not True or args.gpus not in (1, 2, 4) or args.max_steps < 1:
        parser.error('Need successful preparation, 1/2/4 GPUs, and positive max steps')
    if output.exists():
        parser.error('Training output must not exist')
    workspace = prepared/'dreamzero'
    for name, value in report['overlay']['patched_files'].items():
        if digest(workspace/name) != value:
            raise ValueError(f'Generated training source changed: {name}')
    if not report.get('prepared_files'):
        raise ValueError('Prepared dataset fingerprints are missing')
    for name, value in report['prepared_files'].items():
        if digest(prepared/name) != value:
            raise ValueError(f'Prepared data changed: {name}')
    checkpoint, wan, tokenizer = (p.resolve() for p in (args.checkpoint, args.wan, args.tokenizer))
    required = [checkpoint/'config.json', tokenizer/'tokenizer_config.json', wan/'Wan2.1_VAE.pth',
                wan/'models_t5_umt5-xxl-enc-bf16.pth', wan/'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth']
    weight_index = checkpoint/'model.safetensors.index.json'
    if weight_index.is_file():
        index = json.loads(weight_index.read_text())
        shards = set(index.get('weight_map', {}).values())
        if not shards:
            parser.error('Checkpoint weight index is empty')
        for name in shards:
            path = (checkpoint/name).resolve()
            if not path.is_relative_to(checkpoint):
                parser.error('Checkpoint shard path escapes its directory')
            required.append(path)
    else:
        required.append(checkpoint/'model.safetensors')
    if any(not p.is_file() for p in required):
        parser.error('Missing checkpoint/tokenizer/Wan2.1 component files; this entry does not download weights')
    overrides = [
        'data=dreamzero/cruzr_s2_relative', 'model=dreamzero/vla',
        'model/dreamzero/action_head=wan_flow_matching_action_tf', 'model/dreamzero/transform=dreamzero_cotrain',
        'train_architecture=lora', 'num_frames=33', 'action_horizon=24', 'num_views=2',
        'num_frame_per_block=2', 'num_action_per_block=24', 'num_state_per_block=1', 'max_chunk_size=4',
        'frame_seqlen=880', 'max_state_dim=64', 'max_action_dim=32',
        'action_head_cfg.config.action_loss_embodiment_ids=[33]',
        '++action_head_cfg.config.skip_component_loading=true', '++action_head_cfg.config.defer_lora_injection=true',
        f'cruzr_data_root={prepared/"datasets/train"}', f'pretrained_model_path={checkpoint}',
        f'dit_version={wan}', f'text_encoder_pretrained_path={wan/"models_t5_umt5-xxl-enc-bf16.pth"}',
        f'image_encoder_pretrained_path={wan/"models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"}',
        f'vae_pretrained_path={wan/"Wan2.1_VAE.pth"}', f'tokenizer_path={tokenizer}', f'output_dir={output}',
        'image_resolution_width=320', 'image_resolution_height=176', 'bf16=true', 'tf32=true', 'eval_bf16=true',
        'per_device_train_batch_size=1', 'gradient_accumulation_steps=4', 'learning_rate=1e-5',
        f'max_steps={args.max_steps}', 'save_steps=20', 'save_total_limit=5', 'save_lora_only=true',
        'upload_checkpoints=false', 'report_to=none', 'wandb_project=cruzr-local', 'dataloader_num_workers=1',
        'dataloader_pin_memory=false', 'seed=42']
    overrides += [f'training_args.run_name={output.name}', 'ray_num_workers=0']
    if args.deepspeed:
        overrides.append(f'training_args.deepspeed={args.deepspeed.resolve()}')
    entry = str(workspace/'groot/vla/experiment/experiment.py')
    command = ([sys.executable, str(ROOT/'scripts/check_training_config.py'),
                '--workspace', str(workspace), '--output', str(output/'resolved-config.json'), '--'] if args.compose else
               [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={args.gpus}', entry]) + overrides
    print(shlex.join(command))
    if not (args.compose or args.launch):
        print('Plan only. Use --compose first; --launch actually starts training. No 4090 memory claim.')
        return 0
    output.mkdir(parents=True)
    record = {'command': command, 'mode': 'compose' if args.compose else 'train',
              'prepare_report_sha256': digest(prepared/'prepare-report.json'), 'success': False}
    env = os.environ.copy()
    env['PYTHONPATH'] = str(workspace)+os.pathsep+env.get('PYTHONPATH', '')
    env['WANDB_MODE'] = 'disabled'
    try:
        with (output/'training.log').open('w') as log:
            result = subprocess.run(command, cwd=workspace, env=env, stdout=log, stderr=subprocess.STDOUT)
        record.update(success=result.returncode == 0, returncode=result.returncode)
    finally:
        (output/'training-run.json').write_text(json.dumps(record, indent=2)+'\n')
    return 0 if record['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
