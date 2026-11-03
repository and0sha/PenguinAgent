"""Classification correctness and bounded batching without model downloads."""

from contextlib import nullcontext, redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from tools import tracking_config, tracking_siglip as siglip
from tools.tracking_results import TrackingError


class SiglipTests(unittest.TestCase):
    def test_confident_non_penguin_never_becomes_a_penguin_track_label(self):
        for label in ("leopard seal (Hydrurga leptonyx)", "iceberg", "unknown object"):
            with self.subTest(label=label):
                self.assertEqual(siglip._species_decision(label, .99, [.99, .01], [0, 1], ["Adelie", "Emperor"], .1), "NOT_PENGUIN")

    def test_probability_columns_use_classifier_classes_not_column_positions(self):
        self.assertEqual(siglip._species_decision("penguin (Spheniscidae)", .9, [.1, .9], [0, 2],
                                                  ["Adelie", "Chinstrap", "Emperor penguin"], .1), "Emperor_penguin")

    def test_low_confidence_and_explicit_rejection_remain_rejections(self):
        self.assertEqual(siglip._species_decision("penguin (Spheniscidae)", .01, [.99], [0], ["Adelie"], .1), "NOT_PENGUIN")
        self.assertEqual(siglip._species_decision("penguin (Spheniscidae)", .9, [.09], [0], ["Adelie"], .1), "NOT_PENGUIN")
        self.assertEqual(siglip._species_decision("penguin (Spheniscidae)", .9, [.99], [0], ["NOT_PENGUIN"], .1), "NOT_PENGUIN")

    def test_invalid_classifier_output_is_an_error(self):
        for row, classes in (([float("nan")], [0]), ([2.0], [0]), ([.5], [9]), ([.5, .5], [0])):
            with self.subTest(row=row, classes=classes), self.assertRaises(TrackingError):
                siglip._species_decision("penguin (Spheniscidae)", .9, row, classes, ["Adelie"], .1)

    def test_forward_embeddings_are_reused_without_a_second_vision_encoder_call(self):
        class Tensor:
            def __init__(self, values):
                self.values = values
            def cpu(self):
                return self
            def numpy(self):
                return self.values

        class Inputs(dict):
            def to(self, device):
                return self

        model = mock.Mock(return_value=types.SimpleNamespace(logits_per_image=Tensor([[.9]]), image_embeds=Tensor([[.6, .8]])))
        model.get_image_features.side_effect = AssertionError("vision encoder must not run twice")
        processor = mock.Mock(return_value=Inputs(pixel_values="pixels"))
        torch = types.ModuleType("torch")
        torch.inference_mode, torch.sigmoid = lambda: nullcontext(), lambda tensor: tensor
        with mock.patch.dict(sys.modules, {"torch": torch}):
            probabilities, features = siglip._infer_siglip_batch(model, processor, ["image"], ["penguin"], "cpu")
        self.assertEqual((probabilities, features), ([[.9]], [[.6, .8]]))
        model.assert_called_once()
        model.get_image_features.assert_not_called()

    def test_crops_are_loaded_in_bounded_batches_and_closed(self):
        opened = []

        class Image:
            size = (10, 10)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.close()
            def convert(self, mode):
                image = Image()
                opened.append(image)
                return image
            def close(self):
                self.closed = True

        classifier = types.SimpleNamespace(classes_=[0], predict_proba=lambda features: [[1.] for _ in features])
        model = mock.Mock()
        model.to.return_value = model
        torch = types.ModuleType("torch")
        torch.cuda, torch.device = types.SimpleNamespace(is_available=lambda: False), lambda value: value
        transformers = types.ModuleType("transformers")
        transformers.AutoProcessor = types.SimpleNamespace(from_pretrained=mock.Mock(return_value=object()))
        transformers.AutoModel = types.SimpleNamespace(from_pretrained=mock.Mock(return_value=model))
        joblib = types.ModuleType("joblib")
        joblib.load = lambda path: {"classifier": classifier, "label_encoder": types.SimpleNamespace(classes_=["Adelie"])}
        pil = types.ModuleType("PIL")
        pil.Image = types.SimpleNamespace(open=lambda path: Image())
        batch_sizes = []

        def infer(model, processor, images, labels, device):
            batch_sizes.append(len(images))
            return [[.9] + [.01] * (len(labels) - 1) for _ in images], [[1.] for _ in images]

        with tempfile.TemporaryDirectory() as directory:
            batch = []
            for index in range(5):
                path = Path(directory) / f"crop-{index}.jpg"
                path.write_bytes(b"image")
                batch.append((str(path), index))
            with mock.patch.dict(sys.modules, {"torch": torch, "transformers": transformers, "joblib": joblib, "PIL": pil}), \
                    mock.patch.object(tracking_config, "SIGLIP_BATCH_SIZE", 2), \
                    mock.patch.object(siglip, "_infer_siglip_batch", side_effect=infer), redirect_stdout(io.StringIO()):
                result = siglip._classify_siglip_crops(batch, .1)
        self.assertEqual(batch_sizes, [2, 2, 1])
        self.assertEqual(result, {index: "Adelie" for index in range(5)})
        self.assertTrue(all(image.closed for image in opened))


if __name__ == "__main__":
    unittest.main()
