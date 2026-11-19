#!/usr/bin/env python3
"""Resolve candidate Cruzr Hydra configuration without importing Torch/models."""
import argparse
import json
from pathlib import Path


def compose_config(workspace, overrides):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    config_dir = Path(workspace).resolve()/'groot/vla/configs'
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name='conf', overrides=overrides)
    # Resolve interpolations without instantiating models/tokenizers. Upstream
    # trainer arguments and reusable transform templates intentionally contain
    # ??? until runtime, so validate the actually used Cruzr branch separately.
    OmegaConf.resolve(cfg)
    used = {'dataset': cfg.train_dataset, 'transform': cfg.transforms.cruzr_s2,
            'modalities': cfg.modality_configs.cruzr_s2, 'model': cfg.model,
            'training_args': cfg.training_args}
    resolved = OmegaConf.to_container(OmegaConf.create(used), resolve=True, throw_on_missing=True)
    if (cfg.num_views != 2 or cfg.embodiment_tag_to_projector_index.cruzr_s2 != 33 or
            list(cfg.action_head_cfg.config.action_loss_embodiment_ids) != [33] or
            cfg.save_total_limit < 5 or cfg.max_steps < 1):
        raise ValueError('Invalid Cruzr view/projector/loss/checkpoint configuration')
    return cfg, resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('overrides', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    overrides = args.overrides[1:] if args.overrides[:1] == ['--'] else args.overrides
    cfg, resolved = compose_config(args.workspace, overrides)
    args.output.write_text(json.dumps(resolved, ensure_ascii=False, indent=2)+'\n')
    print(f'Cruzr configuration resolved: {args.output}')
    print('No model instantiation, weight loading, CUDA allocation or training was performed.')


if __name__ == '__main__':
    main()
