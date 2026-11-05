"""NVIDIA NIM generator: request shape, caching, retries and key handling. No network or API key needed."""
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from analysis.rag import nim

KEY = "nvapi-test-not-a-real-key"
MESSAGES = [{"role": "system", "content": "Answer from the sources."}, {"role": "user", "content": "What is mill scale?"}]


def reply(text):
    return {"choices": [{"message": {"role": "assistant", "content": f"  {text}  "}}]}


class NimSpecTest(unittest.TestCase):
    def test_disabled_without_key_or_model(self):
        self.assertIsNone(nim.nim_spec({}))
        self.assertIsNone(nim.nim_spec({"NVIDIA_API_KEY": KEY}))
        self.assertIsNone(nim.nim_spec({"NIM_MODEL": "nvidia/some-nemotron"}))

    def test_spec_names_the_model_and_never_the_key(self):
        spec = nim.nim_spec({"NVIDIA_API_KEY": KEY, "NIM_MODEL": "nvidia/some-nemotron"})
        self.assertEqual(spec["name"], "nvidia/some-nemotron")
        self.assertEqual(spec["provider"], "nim")
        self.assertNotIn(KEY, json.dumps(spec))


class NimGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name) / "nim.json"
        self.logs = []

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, **kw):
        return nim.NimGenerator("nvidia/some-nemotron", self.cache, api_key=KEY, log=self.logs.append,
                                sleep=lambda s: None, **kw)

    def test_requires_a_key(self):
        with mock.patch.dict("os.environ", {}, clear=True), self.assertRaises(RuntimeError):
            nim.NimGenerator("m", self.cache)

    def test_greedy_request_and_stripped_answer(self):
        with mock.patch.object(nim, "_request", return_value=reply("Mill scale is an oxide layer.")) as req:
            out = self.make().generate([MESSAGES], max_new_tokens=220)
        self.assertEqual(out, ["Mill scale is an oxide layer."])
        path, payload, api_key, _ = req.call_args.args
        self.assertEqual(path, "/chat/completions")
        self.assertEqual(api_key, KEY)
        self.assertEqual(payload["model"], "nvidia/some-nemotron")
        self.assertEqual(payload["messages"], MESSAGES)
        self.assertEqual((payload["max_tokens"], payload["temperature"], payload["stream"]), (220, 0, False))

    def test_cache_avoids_repeat_calls_and_holds_no_key(self):
        with mock.patch.object(nim, "_request", return_value=reply("A")) as req:
            self.make().generate([MESSAGES, MESSAGES])
            self.make().generate([MESSAGES])  # a new instance reads the cache file
        self.assertEqual(req.call_count, 1)
        self.assertNotIn(KEY, self.cache.read_text(encoding="utf-8"))

    def test_retries_rate_limit_then_succeeds(self):
        busy = urllib.error.HTTPError("u", 429, "Too Many Requests", {}, io.BytesIO(b""))
        with mock.patch.object(nim, "_request", side_effect=[busy, busy, reply("ok")]) as req:
            out = self.make(retries=3).generate([MESSAGES])
        self.assertEqual(out, ["ok"])
        self.assertEqual(req.call_count, 3)
        self.assertTrue(all(KEY not in line for line in self.logs))

    def test_non_retryable_error_stops_without_leaking_the_key(self):
        denied = urllib.error.HTTPError("u", 401, "Unauthorized", {}, io.BytesIO(b""))
        with mock.patch.object(nim, "_request", side_effect=denied), self.assertRaises(RuntimeError) as ctx:
            self.make().generate([MESSAGES])
        self.assertIn("401", str(ctx.exception))
        self.assertNotIn(KEY, str(ctx.exception))
        self.assertFalse(self.cache.exists())


if __name__ == "__main__":
    unittest.main()
