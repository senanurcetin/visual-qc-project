import unittest

from analysis.dl.preflight import (
    EXPECTED_CLASSES,
    MIN_FREE_BYTES,
    Check,
    check_dataset_layout,
    evaluate,
    exit_code,
    label_of,
    render,
)


def good_files():
    return [f"{c}_{i}.jpg" for c in EXPECTED_CLASSES for i in range(1, 301)]


def good_facts(**overrides):
    facts = {
        "torch_version": "2.5.0", "cuda_device": "NVIDIA A10", "cuda_memory_gb": 24.0,
        "free_bytes": 50 * 1024**3, "dataset_files": good_files(), "unreadable_samples": [],
        "weights_cached": True, "baseline_present": True, "outputs_writable": True, "zip_path": "/x/NEU-CLS.zip",
    }
    facts.update(overrides)
    return facts


def by_name(checks):
    return {c.name: c for c in checks}


class LayoutTests(unittest.TestCase):
    def test_labels_keep_underscores_in_class_names(self):
        self.assertEqual(label_of("pitted_surface_12.jpg"), "pitted_surface")
        self.assertEqual(label_of("rolled-in_scale_300.jpg"), "rolled-in_scale")

    def test_complete_dataset_passes(self):
        self.assertEqual(check_dataset_layout(good_files()).status, "PASS")

    def test_missing_images_and_unknown_classes_fail_with_specifics(self):
        files = [f for f in good_files() if f != "scratches_7.jpg"] + ["mystery_1.jpg"]
        check = check_dataset_layout(files)
        self.assertEqual(check.status, "FAIL")
        self.assertIn("scratches: found 299, expected 300", check.detail)
        self.assertIn("unexpected class 'mystery'", check.detail)


class EvaluateTests(unittest.TestCase):
    def test_everything_in_place_is_ready(self):
        checks = evaluate(good_facts())
        self.assertEqual(exit_code(checks), 0)
        self.assertTrue(all(c.status == "PASS" for c in checks))
        self.assertIn("Preflight: ready", render(checks))

    def test_no_gpu_warns_but_does_not_block(self):
        checks = evaluate(good_facts(cuda_device=None))
        self.assertEqual(by_name(checks)["gpu"].status, "WARN")
        self.assertEqual(exit_code(checks), 0)
        self.assertIn("with warnings", render(checks))

    def test_missing_torch_blocks(self):
        checks = evaluate(good_facts(torch_error="ModuleNotFoundError: torch"))
        self.assertEqual(by_name(checks)["torch"].status, "FAIL")
        self.assertEqual(exit_code(checks), 1)

    def test_low_disk_blocks(self):
        checks = evaluate(good_facts(free_bytes=MIN_FREE_BYTES - 1))
        self.assertEqual(by_name(checks)["disk"].status, "FAIL")

    def test_unreachable_dataset_blocks_and_says_where_to_put_the_zip(self):
        checks = evaluate(good_facts(dataset_files=None, dataset_url_ok=False, dataset_url_detail="URLError"))
        check = by_name(checks)["dataset"]
        self.assertEqual(check.status, "FAIL")
        self.assertIn("/x/NEU-CLS.zip", check.detail)
        self.assertIn("NOT ready", render(checks))

    def test_dataset_that_can_be_downloaded_or_is_zipped_is_acceptable(self):
        self.assertEqual(by_name(evaluate(good_facts(dataset_files=None, dataset_url_ok=True)))["dataset"].status, "PASS")
        self.assertEqual(by_name(evaluate(good_facts(dataset_files=None, zip_present=True)))["dataset"].status, "WARN")

    def test_undecodable_images_block(self):
        checks = evaluate(good_facts(unreadable_samples=["crazing_1.jpg"]))
        self.assertEqual(by_name(checks)["dataset images"].status, "FAIL")

    def test_unreachable_weights_only_warn(self):
        checks = evaluate(good_facts(weights_cached=False, weights_url_ok=False))
        self.assertEqual(by_name(checks)["pretrained weights"].status, "WARN")
        self.assertEqual(exit_code(checks), 0)

    def test_exit_code_and_render_follow_the_worst_status(self):
        self.assertEqual(exit_code([Check("PASS", "a", ""), Check("WARN", "b", "")]), 0)
        self.assertEqual(exit_code([Check("WARN", "b", ""), Check("FAIL", "c", "")]), 1)


if __name__ == "__main__":
    unittest.main()
