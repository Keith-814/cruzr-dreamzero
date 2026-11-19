#!/usr/bin/env python3
"""Decode a saved camera image without opening ROS or moving a robot."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--scanner', choices=('opencv-qr', 'zbar'), default='zbar')
    parser.add_argument('--expected-code')
    args = parser.parse_args()
    import cv2
    from cruzr_sim.scanning import ImageScanner
    frame = cv2.imread(str(args.image))
    if frame is None:
        parser.error('Cannot read image')
    result = ImageScanner(args.scanner)({'id': str(args.image),
                                        'images': {'saved_camera': cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)}})
    texts = {row['text'] for row in result['codes']}
    result['success'] = args.expected_code in texts if args.expected_code else len(texts) == 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
