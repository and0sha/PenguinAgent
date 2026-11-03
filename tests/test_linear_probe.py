"""Exercise dataset alignment with real image decoding and a small fake encoder."""
from contextlib import nullcontext, redirect_stdout
import io
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from PIL import Image
from tools.train_linear_probe import extract_features


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def norm(self, **kwargs):
        return np.linalg.norm(self.values, axis=-1, keepdims=True)

    def __truediv__(self, other):
        with np.errstate(invalid="ignore", divide="ignore"):
            return Tensor(self.values / other)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class Inputs(dict):
    def to(self, device):
        return self


class Model:
    def parameters(self):
        return iter([types.SimpleNamespace(device="cpu")])

    def get_image_features(self, pixels):
        return Tensor(pixels)


class LinearProbeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in ("a", "b", "c"):
            (self.root / name).mkdir()
        self.patch = mock.patch.dict("sys.modules", {
            "torch": types.SimpleNamespace(no_grad=nullcontext),
            "tqdm": types.SimpleNamespace(tqdm=lambda iterable, **kwargs: iterable),
        })
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def image(self, path, color):
        with Image.new("RGB", (2, 2), color) as image:
            image.save(self.root / path)

    @staticmethod
    def processor(images, **kwargs):
        return Inputs(pixels=[image.getpixel((0, 0))[:2] for image in images])

    def extract(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return extract_features(self.root, self.processor, kwargs.pop("model", Model()), **kwargs)

    def test_bad_images_do_not_shift_labels_across_classes_or_batches(self):
        (self.root / "a/0.png").write_bytes(b"corrupt image")
        self.image("b/0.png", (255, 0, 0))
        self.image("c/0.png", (0, 255, 0))
        features, labels = self.extract(batch_size=2)
        self.assertEqual(labels.tolist(), ["b", "c"])
        np.testing.assert_array_equal(features, [[1, 0], [0, 1]])

    def test_all_corrupt_images_fail_with_actionable_error(self):
        (self.root / "a/0.png").write_bytes(b"corrupt image")
        with self.assertRaisesRegex(ValueError, "No readable images"):
            self.extract()

    def test_model_cannot_return_a_different_number_of_features(self):
        self.image("a/0.png", (255, 0, 0))
        model = Model()
        model.get_image_features = lambda **kwargs: Tensor([[1, 0], [0, 1]])
        with self.assertRaisesRegex(ValueError, "misaligned"):
            self.extract(model=model)

    def test_zero_norm_embedding_is_rejected(self):
        self.image("a/0.png", (0, 0, 0))
        with self.assertRaisesRegex(ValueError, "invalid"):
            self.extract()

    def test_invalid_batch_size_is_rejected(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "batch_size"):
                self.extract(batch_size=value)


if __name__ == "__main__":
    unittest.main()
