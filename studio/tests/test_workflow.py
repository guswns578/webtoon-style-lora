"""Offline workflow regressions; run with python -m unittest discover -s tests."""

import json, sys, tempfile, unittest, zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine as e
import app
from PIL import Image, ImageDraw


class Workflow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = patch.object(e, "DB", self.root / "jobs.sqlite3")
        self.db.start()
        self.results = patch.object(e, "RESULTS", self.root / "results")
        self.results.start()
        e.RESULTS.mkdir()
        e.init_db()
        self.params = {
            "character": "char_a",
            "seed": 42,
            "shots": "both",
            "foreground": False,
            "background_mode": "white",
            "b_height": 125,
            "c_height": 33.6,
        }

    def tearDown(self):
        self.results.stop()
        self.db.stop()
        self.tmp.cleanup()

    def test_queue_cancel_recovery_and_seed(self):
        p = dict(self.params, seed=-1)
        first = e.enqueue(p)
        second = e.enqueue(self.params)
        self.assertGreaterEqual(json.loads(e.job(first)["params"])["seed"], 0)
        app.cancel(second)
        self.assertEqual(e.job(second)["status"], "cancelled")
        e.update(first, status="running")
        e.init_db()
        self.assertEqual(e.job(first)["status"], "failed")

    def test_masks_composition_review_export(self):
        raw = Image.open(e.ROOT / "examples/raw.png").convert("RGB")
        fm = Image.open(e.ROOT / "examples/figure_mask.png").convert("L")
        om = Image.open(e.ROOT / "examples/omission_mask.png").convert("L")
        ident = e.enqueue(self.params)
        result = e.compose_files(raw, self.params, e.RESULTS / ident, fm, om)
        with Image.open(result["B"]) as image:
            self.assertEqual(image.size, (1024, 1280))
        with Image.open(result["C_cutout"]) as image:
            self.assertEqual(image.mode, "RGBA")
        self.assertLess(result["layout"]["C"]["height_fraction"], 0.34)
        self.assertEqual(
            app.editor_mask(app.editor_value(raw, om), raw.size).tobytes(), om.tobytes()
        )
        e.update(ident, status="review", result=json.dumps(result))
        app.save_review(ident, app.editor_value(raw, fm), app.editor_value(raw, om))
        revised = json.loads(e.job(ident)["result"])
        self.assertTrue(revised["review_revision"])
        self.assertIn("revisions", revised["raw"])
        app.decision(ident, "approved")
        archive = app.export(ident)
        with zipfile.ZipFile(archive) as z:
            self.assertIsNone(z.testzip())
            self.assertEqual(json.loads(z.read("job.json"))["status"], "approved")
            self.assertEqual(z.read("C.png"), Path(revised["C"]).read_bytes())

    def test_manual_mask_recovers_failed_isolation(self):
        raw = Image.new("RGB", (896, 1152), "white")
        ident = e.enqueue(self.params)
        folder = e.RESULTS / ident
        folder.mkdir()
        raw.save(folder / "generated_raw.png")
        e.update(ident, status="failed", error="separation failed")
        gallery, _, _, _, current = app.load_job(ident)
        self.assertTrue(gallery)
        self.assertEqual(current, ident)
        fm = Image.new("L", raw.size)
        ImageDraw.Draw(fm).rectangle((300, 100, 550, 1050), fill=255)
        app.save_review(
            ident, app.editor_value(raw, fm), app.editor_value(raw, Image.new("L", raw.size))
        )
        self.assertEqual(e.job(ident)["status"], "review")

    def test_face_failure_requires_review_and_all_poses(self):
        raw = Image.open(e.ROOT / "examples/raw.png")
        with patch.object(e, "face_mask", side_effect=RuntimeError("test")):
            result = e.compose_files(raw, self.params, self.root / "face_failure")
        self.assertTrue(result["warnings"])
        self.assertIsNone(result["face"])
        for alias in e.PRESETS["characters"]:
            image, text = e.pose_image(alias, "recommended")
            self.assertEqual(image.size, (896, 1152))
            self.assertTrue(text)
            self.assertIsNotNone(image.getbbox())

    def test_remote_address_refused(self):
        with self.assertRaises(ValueError):
            e.save_settings("https://example.com", "x", "x")

    def test_transparent_masks_remain_full_strength(self):
        raw = Image.new("RGB", (80, 100), "black")
        mask = Image.new("L", raw.size)
        ImageDraw.Draw(mask).rectangle((20, 30, 50, 60), fill=255)
        value = app.editor_value(raw, mask)
        alpha = value["layers"][0].getchannel("A")
        self.assertEqual(alpha.getextrema(), (0, 100))
        self.assertEqual(app.editor_mask(value, raw.size).tobytes(), mask.tobytes())
        self.assertLess(value["composite"].getpixel((25, 35))[2], 255)
        value["layers"][0].putpixel((25, 35), (0, 0, 0, 0))
        self.assertEqual(app.editor_mask(value, raw.size).getpixel((25, 35)), 0)

    def test_simple_export_and_view_match_current_revision(self):
        raw = Image.open(e.ROOT / "examples/raw.png").convert("RGB")
        fm = Image.open(e.ROOT / "examples/figure_mask.png").convert("L")
        om = Image.open(e.ROOT / "examples/omission_mask.png").convert("L")
        ident = e.enqueue(self.params)
        result = e.compose_files(raw, self.params, e.RESULTS / ident, fm, om)
        e.update(ident, status="review", result=json.dumps(result))
        view = app.load_view(ident)
        self.assertEqual(view[1], result["B"])
        self.assertEqual(view[2], result["C"])
        self.assertIsNone(view[-1])
        with zipfile.ZipFile(app.export(ident, False)) as z:
            self.assertEqual(set(z.namelist()), {"B.png", "C.png"})

    def test_remove_comparison_preserves_character_masks_and_export(self):
        raw = Image.open(e.ROOT / "examples/raw.png").convert("RGB")
        fm = Image.open(e.ROOT / "examples/figure_mask.png").convert("L")
        om = Image.open(e.ROOT / "examples/omission_mask.png").convert("L")
        p = {**self.params, "foreground": True}
        ident = e.enqueue(p)
        result = e.compose_files(raw, p, e.RESULTS / ident, fm, om)
        e.update(ident, status="approved", result=json.dumps(result))
        before = Path(result["B"]).read_bytes()
        with patch.object(e, "api", side_effect=AssertionError("No GPU request allowed")):
            app.remove_comparison(ident)
        j = e.job(ident)
        clean = json.loads(j["result"])
        self.assertFalse(json.loads(j["params"])["foreground"])
        self.assertEqual(j["status"], "review")
        self.assertTrue(clean["comparison_removed"])
        self.assertNotEqual(Path(clean["B"]).read_bytes(), before)
        for key in ["figure_mask", "omission_mask"]:
            with Image.open(clean[key]) as saved, self.subTest(key=key):
                expected = fm if key == "figure_mask" else om
                self.assertEqual(saved.tobytes(), expected.tobytes())
        with Image.open(clean["raw"]) as saved:
            self.assertEqual(saved.convert("RGB").tobytes(), raw.tobytes())
        expected, _, _ = e.render(e.isolate(raw), "B", {**p, "foreground": False})
        with Image.open(clean["B"]) as saved:
            self.assertEqual(saved.tobytes(), expected.tobytes())
        with zipfile.ZipFile(app.export(ident, False)) as z:
            self.assertEqual(z.read("B.png"), Path(clean["B"]).read_bytes())
        app.remove_comparison(ident)
        self.assertEqual(json.loads(e.job(ident)["result"])["B"], clean["B"])

    def test_export_automatically_removes_foreground_and_refreshes_view(self):
        raw = Image.open(e.ROOT / "examples/raw.png").convert("RGB")
        fm = Image.open(e.ROOT / "examples/figure_mask.png").convert("L")
        om = Image.open(e.ROOT / "examples/omission_mask.png").convert("L")
        params = {**self.params, "foreground": True}
        ident = e.enqueue(params)
        result = e.compose_files(raw, params, e.RESULTS / ident, fm, om)
        e.update(ident, status="review", result=json.dumps(result))
        view = app.export_view(ident, False)
        clean = json.loads(e.job(ident)["result"])
        self.assertFalse(json.loads(e.job(ident)["params"])["foreground"])
        self.assertEqual(view[1], clean["B"])
        self.assertTrue(view[-1].endswith(".zip"))
        with zipfile.ZipFile(view[-1]) as z:
            self.assertEqual(z.read("C.png"), Path(clean["C"]).read_bytes())

    def test_grid_only_is_removed_on_export(self):
        raw = Image.open(e.ROOT / "examples/raw.png").convert("RGB")
        fm = Image.open(e.ROOT / "examples/figure_mask.png").convert("L")
        om = Image.open(e.ROOT / "examples/omission_mask.png").convert("L")
        params = {**self.params, "background_mode": "guide", "foreground": False}
        ident = e.enqueue(params)
        result = e.compose_files(raw, params, e.RESULTS / ident, fm, om)
        e.update(ident, status="review", result=json.dumps(result))
        original = Path(result["C"]).read_bytes()
        app.export_view(ident, False)
        clean = json.loads(e.job(ident)["result"])
        self.assertEqual(json.loads(e.job(ident)["params"])["background_mode"], "white")
        self.assertTrue(clean["guide_removed"])
        self.assertNotEqual(original, Path(clean["C"]).read_bytes())
        with Image.open(clean["C"]) as image:
            self.assertEqual(image.convert("RGB").getpixel((100, 445)), (255, 255, 255))

    def test_uploaded_background_is_preserved_when_removing_foreground(self):
        background = self.root / "uploaded.png"
        Image.new("RGB", (1024, 1280), (90, 120, 180)).save(background)
        raw = Image.open(e.ROOT / "examples/raw.png").convert("RGB")
        fm = Image.open(e.ROOT / "examples/figure_mask.png").convert("L")
        om = Image.open(e.ROOT / "examples/omission_mask.png").convert("L")
        params = {
            **self.params,
            "background_mode": "upload",
            "background_path": str(background),
            "foreground": True,
        }
        ident = e.enqueue(params)
        result = e.compose_files(raw, params, e.RESULTS / ident, fm, om)
        e.update(ident, status="review", result=json.dumps(result))
        app.remove_comparison(ident)
        clean = json.loads(e.job(ident)["result"])
        self.assertEqual(json.loads(e.job(ident)["params"])["background_path"], str(background))
        with Image.open(clean["C"]) as image:
            self.assertEqual(image.convert("RGB").getpixel((100, 445)), (90, 120, 180))

    def test_preview_sizes_seed_modes_and_completion(self):
        from ui_helpers import seed_value

        self.assertEqual(seed_value("verified", 123), -2)
        self.assertEqual(seed_value("random", 123), -1)
        self.assertEqual(seed_value("manual", 123), 123)
        with self.assertRaises(ValueError):
            seed_value("manual", -3)
        b, c, _ = app.layout_preview(125, 33.6, "guide", None, True)
        _, larger, _ = app.layout_preview(125, 50, "guide", None, True)
        self.assertEqual(b.size, (1024, 1280))
        self.assertNotEqual(c.tobytes(), larger.tobytes())
        ident = e.enqueue(self.params)
        result = {
            "raw": str(e.ROOT / "examples/raw.png"),
            "B": str(e.ROOT / "examples/B.png"),
            "C": str(e.ROOT / "examples/C.png"),
            "figure_mask": str(e.ROOT / "examples/figure_mask.png"),
            "omission_mask": str(e.ROOT / "examples/omission_mask.png"),
        }
        e.update(ident, status="review", result=json.dumps(result))
        batch = {"ids": [ident], "opened": False}
        output = app.poll_view(ident, batch, True, None)
        self.assertTrue(output[3]["opened"])
        self.assertEqual(output[-1]["selected"], "review")
        self.assertEqual(output[-3], ident)
        again = app.poll_view(ident, output[3], True, ident)
        self.assertEqual(again[-1], {"__type__": "update"})
        off = app.poll_view(ident, batch, False, None)
        self.assertFalse(off[3]["opened"])


if __name__ == "__main__":
    unittest.main()
