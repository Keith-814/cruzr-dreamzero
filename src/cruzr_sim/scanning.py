"""Image decoders with explicit provenance; fixture results are simulation only."""
from pathlib import Path


class ImageScanner:
    def __init__(self, kind='opencv-qr', output=None):
        if kind not in ('opencv-qr', 'zbar'):
            raise ValueError('Choose opencv-qr or zbar')
        self.kind, self.output, self.index = kind, Path(output) if output else None, 0
        import cv2
        self.cv2 = cv2
        self.detector = cv2.QRCodeDetector() if kind == 'opencv-qr' else None
        if kind == 'zbar':
            # Requires libzbar in the target OS in addition to the Python extra.
            from pyzbar.pyzbar import decode
            self.decode = decode

    def __call__(self, capture):
        codes = []
        self.index += 1
        for name, image in capture['images'].items():
            import numpy as np
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError('Scanner requires uint8 HWC RGB')
            if not name.replace('_', '').isalnum():
                raise ValueError('Invalid camera key')
            bgr = self.cv2.cvtColor(image, self.cv2.COLOR_RGB2BGR)
            if self.output:
                self.output.mkdir(parents=True, exist_ok=True)
                if not self.cv2.imwrite(str(self.output/f'{self.index:03d}-{name}.png'), bgr):
                    raise OSError('Cannot save scan evidence')
            if self.kind == 'opencv-qr':
                ok, values, _, _ = self.detector.detectAndDecodeMulti(bgr)
                if not ok:
                    value, _, _ = self.detector.detectAndDecode(bgr)
                    values = [value] if value else []
                codes.extend({'text': value, 'format': 'QR', 'camera': name} for value in values if value)
            else:
                codes.extend({'text': item.data.decode('utf-8'), 'format': item.type, 'camera': name}
                             for item in self.decode(bgr))
        return {'codes': codes, 'source': 'camera', 'decoder': self.kind}


class FixtureScanner:
    """Tests motion sequencing; never represents successful physical decoding."""
    def __init__(self, code):
        if not isinstance(code, str) or not code.strip():
            raise ValueError('Fixture code must be nonempty')
        self.code = code

    def __call__(self, capture):
        return {'codes': [{'text': self.code, 'format': 'FIXTURE'}], 'source': 'fixture'}
