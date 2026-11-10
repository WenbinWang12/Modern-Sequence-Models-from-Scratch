"""在远端直接运行的标准库检查，核对教学产物及真实输入边界。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

BOOK = Path(__file__).resolve().parent
ROOT = BOOK.parents[1]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, BOOK / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ACQUISITION = load_module("acquisition_walkthrough")
SPLIT = load_module("split_walkthrough")
CLEANING = load_module("cleaning_walkthrough")


def read_jsonl(path):
    # 记录边界只认 LF，不能把正文中的 U+2028 拆成额外记录。
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def snapshot(directory):
    return {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}


class DataArtifactChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="book-data-check-")
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)

    def clean_args(self, source, output_name):
        return argparse.Namespace(
            input=source,
            output=self.work / output_name,
            min_prose_chars=12,
            min_code_chars=1,
            max_chars=120,
            max_duplicate_line_fraction=0.6,
        )

    def write_records(self, name, records, *, final_newline=True):
        source = self.work / name
        text = "\n".join(json.dumps(row, ensure_ascii=False) for row in records)
        source.write_bytes((text + ("\n" if final_newline else "")).encode("utf-8"))
        return source

    def check_acquisition_artifacts(self, source, output):
        before = source.read_bytes()
        returned = ACQUISITION.run(source, output)
        report = json.loads((output / "source-record.json").read_text(encoding="utf-8"))
        self.assertEqual(returned, report)
        original = output / report["original"]["file"]
        variant = output / report["variant"]["file"]
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(original.read_bytes(), before)
        self.assertEqual(variant.read_bytes(), before + b"\n")
        for key, path in (("original", original), ("variant", variant)):
            data = path.read_bytes()
            self.assertEqual(report[key]["bytes"], len(data))
            self.assertEqual(report[key]["sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(read_jsonl(original), read_jsonl(variant))
        self.assertEqual(report["records"], len(read_jsonl(original)))
        self.assertEqual(report["byte_counts_differ_by"], 1)
        self.assertTrue(report["same_parsed_records"])
        self.assertTrue(report["different_byte_digest"])
        self.assertTrue(report["original_preserved"])
        self.assertEqual(report["source"]["terms_review"], "not_performed_by_this_script")
        return report

    def test_acquisition_default_saved_objects_and_no_overwrite(self):
        source = ROOT / "examples" / "sample.jsonl"
        output = self.work / "acquisition"
        report = self.check_acquisition_artifacts(source, output)
        self.assertEqual(report["records"], 4)
        saved = snapshot(output)
        with self.assertRaises(FileExistsError):
            ACQUISITION.run(source, output)
        self.assertEqual(snapshot(output), saved)

    def test_acquisition_no_final_lf_and_unicode_separator(self):
        source = self.write_records(
            "source.jsonl",
            [{"id": "one", "text": "Before\u2028after."}],
            final_newline=False,
        )
        self.assertFalse(source.read_bytes().endswith(b"\n"))
        self.assertIn("\u2028".encode("utf-8"), source.read_bytes())
        report = self.check_acquisition_artifacts(source, self.work / "acquisition-unicode")
        self.assertEqual(report["records"], 1)
        saved = read_jsonl(self.work / "acquisition-unicode" / "with-extra-newline.jsonl")
        self.assertEqual(saved[0]["text"], "Before\u2028after.")

    def test_split_artifact_conservation_groups_probes_and_order(self):
        source = BOOK / "fixtures" / "split_records.jsonl"
        records = read_jsonl(source)
        output = self.work / "split"
        returned = SPLIT.run(source, output, 17)
        report = json.loads((output / "split-report.json").read_text(encoding="utf-8"))
        self.assertEqual(returned, report)
        self.assertEqual(report["input_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
        seen, group_membership, content_membership = {}, {}, {}
        for split in ("train", "validation", "test"):
            path = output / f"{split}.jsonl"
            subset = read_jsonl(path)
            self.assertEqual(report["totals"][split]["documents"], len(subset))
            self.assertEqual(report["totals"][split]["groups"], len({r["split_group"] for r in subset}))
            self.assertEqual(
                report["totals"][split]["utf8_content_bytes"],
                sum(len(record["text"].encode("utf-8")) for record in subset),
            )
            self.assertEqual(report["totals"][split]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            for record in subset:
                self.assertNotIn(record["id"], seen)
                seen[record["id"]] = (split, record)
                group_membership.setdefault(record["split_group"], set()).add(split)
                content_membership.setdefault(record["text"], set()).add(split)
        self.assertEqual(len(seen), 12)
        self.assertEqual(len(group_membership), 10)
        self.assertEqual({key: value[1] for key, value in seen.items()}, {r["id"]: r for r in records})
        self.assertTrue(all(len(memberships) == 1 for memberships in group_membership.values()))
        self.assertTrue(all(len(memberships) == 1 for memberships in content_membership.values()))
        self.assertEqual(
            {probe["expected"] for probe in report["failure_probes"]},
            {"group_leak", "normalized_content_leak"},
        )
        self.assertTrue(all(probe["rejected"] for probe in report["failure_probes"]))
        self.assertTrue(report["order_independent"])
        self.assertTrue(report["existing_assignments_stable_after_append"])

        # 用新的实际输入文件再跑一遍，而不只相信报告里的布尔值。
        reversed_source = self.write_records("reversed.jsonl", list(reversed(records)))
        reversed_output = self.work / "split-reversed"
        SPLIT.run(reversed_source, reversed_output, 17)
        for split in ("train", "validation", "test"):
            self.assertEqual(
                read_jsonl(output / f"{split}.jsonl"),
                read_jsonl(reversed_output / f"{split}.jsonl"),
            )
        extra = {"id": "extra", "split_group": "new-independent-group",
                 "source": "test", "text": "An independent new document."}
        appended_source = self.write_records("appended.jsonl", records + [extra])
        appended_output = self.work / "split-appended"
        SPLIT.run(appended_source, appended_output, 17)
        for split in ("train", "validation", "test"):
            before_ids = {r["id"] for r in read_jsonl(output / f"{split}.jsonl")}
            after_ids = {r["id"] for r in read_jsonl(appended_output / f"{split}.jsonl") if r["id"] != "extra"}
            self.assertEqual(before_ids, after_ids)
        saved = snapshot(output)
        with self.assertRaises(FileExistsError):
            SPLIT.run(source, output, 17)
        self.assertEqual(snapshot(output), saved)

    def test_cleaning_fixture_counts_diagnostics_and_indentation(self):
        args = self.clean_args(BOOK / "fixtures" / "cleaning_records.jsonl", "clean")
        returned = CLEANING.run(args)
        report = json.loads((args.output / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(returned, report)
        kept = read_jsonl(args.output / "kept.jsonl")
        rejected = read_jsonl(args.output / "rejected.jsonl")
        self.assertEqual((report["input_records"], len(kept), len(rejected)), (13, 5, 8))
        self.assertEqual((report["kept_records"], report["rejected_records"]), (5, 8))
        self.assertEqual(sum(report["primary_counts"].values()), 13)
        self.assertEqual({r["id"] for r in kept}, {"c01", "c02", "c03", "c12", "c13"})
        by_id = {r["id"]: r for r in kept}
        self.assertEqual(by_id["c02"]["text"], "    return x + 1\n")
        self.assertEqual(by_id["c13"]["text"], "    pass\n    pass\n    pass\n")
        self.assertEqual(by_id["c01"]["text"], "Café by the sea.")
        self.assertEqual(by_id["c01"]["original_text"], "Cafe\u0301 by the sea.")
        decisions = {r["id"]: r for r in report["decisions"]}
        self.assertEqual(decisions["c04"]["primary_reason"], "empty")
        self.assertEqual(set(decisions["c04"]["diagnostics"]), {"empty", "too_short"})
        self.assertEqual(decisions["c06"]["primary_reason"], "too_short")
        self.assertEqual(set(decisions["c06"]["diagnostics"]), {"too_short", "repeated_line"})
        self.assertGreater(sum(report["diagnostic_counts"].values()), report["rejected_records"])
        self.assertEqual(report["stages"][-1]["remaining"], len(kept))
        saved = snapshot(args.output)
        with self.assertRaises(FileExistsError):
            CLEANING.run(args)
        self.assertEqual(snapshot(args.output), saved)

    def test_cleaning_duplicate_threshold_includes_exact_boundary(self):
        cases = (
            ("below", 0.19999999999999998, "A.\nB.\nC.\nD.\nA.", "repeated_line", 1),
            ("exact", 0.2, "A.\nB.\nC.\nD.\nA.", "repeated_line", 1),
            ("above", 0.20000000000000004, "A.\nB.\nC.\nD.\nA.", "kept", 1),
            ("zero", 0.0, "A.\nB.\nC.\nD.\nE.", "repeated_line", 0),
            ("one", 1.0, "A.\nA.\nA.\nA.\nA.", "kept", 4),
        )
        for name, threshold, body, reason, repeats in cases:
            with self.subTest(name=name):
                source = self.write_records(f"boundary-{name}.jsonl", [
                    {"id": name, "source": "test", "language": "en", "kind": "prose", "text": body}
                ])
                args = self.clean_args(source, f"clean-boundary-{name}")
                args.max_duplicate_line_fraction = threshold
                report = CLEANING.run(args)
                decision = report["decisions"][0]
                self.assertEqual(decision["primary_reason"], reason)
                self.assertEqual(decision["metrics"]["nonempty_lines"], 5)
                self.assertEqual(decision["metrics"]["duplicate_comparison_lines"], repeats)
                self.assertEqual(decision["metrics"]["duplicate_line_fraction"], repeats / 5)
                self.assertEqual(report["parameters"]["duplicate_line_threshold_decimal"], str(threshold))
                filename = "kept.jsonl" if reason == "kept" else "rejected.jsonl"
                self.assertEqual(len(read_jsonl(args.output / filename)), 1)
                self.assertEqual(report["kept_records"], int(reason == "kept"))

    def test_duplicate_json_members_are_not_silently_overwritten(self):
        base = {"id": "dup", "source": "test", "language": "en", "kind": "prose",
                "split_group": "project-a", "text": "This text is otherwise acceptable."}
        extras = (
            '"text": 42',
            '"te\\u0078t": "A different sufficiently long text."',
            '"split_group": "project-b"',
            '"metadata": {"score": 1, "score": 2}',
            '"metadata": {"items": [{"tag": "a", "tag": "b"}]}',
        )
        for index, extra in enumerate(extras):
            with self.subTest(extra=extra):
                raw_line = json.dumps(base)[:-1] + ", " + extra + "}"
                source = self.work / f"duplicate-member-{index}.jsonl"
                source.write_bytes((raw_line + "\n").encode("utf-8"))
                acquired = self.work / f"duplicate-acquisition-{index}"
                split = self.work / f"duplicate-split-{index}"
                with self.assertRaisesRegex(ValueError, "duplicate_json_key"):
                    ACQUISITION.run(source, acquired)
                with self.assertRaisesRegex(ValueError, "duplicate_json_key"):
                    SPLIT.run(source, split, 17)
                self.assertFalse(acquired.exists())
                self.assertFalse(split.exists())
                args = self.clean_args(source, f"duplicate-clean-{index}")
                report = CLEANING.run(args)
                self.assertEqual((report["kept_records"], report["rejected_records"]), (0, 1))
                self.assertEqual(report["primary_counts"]["invalid_json"], 1)
                rejected = read_jsonl(args.output / "rejected.jsonl")
                self.assertEqual(rejected[0]["raw_line"], raw_line)
                self.assertIn("duplicate_json_key", rejected[0]["json_error"]["message"])

    def test_split_rejects_nonfinite_metadata_and_preserves_finite_values(self):
        records = read_jsonl(BOOK / "fixtures" / "split_records.jsonl")
        cases = (
            "NaN", "Infinity", "-Infinity", "1e999", "-1e999",
            '{"scores": [{"value": 1e999}]}', '{"scores": [{"value": -1e999}]}',
        )
        for index, literal in enumerate(cases):
            with self.subTest(literal=literal):
                lines = [json.dumps(record, ensure_ascii=False) for record in records]
                lines[0] = lines[0][:-1] + ', "metadata": ' + literal + "}"
                source = self.work / f"split-nonfinite-{index}.jsonl"
                source.write_text("\n".join(lines) + "\n", encoding="utf-8")
                output = self.work / f"split-nonfinite-{index}"
                with self.assertRaises(ValueError):
                    SPLIT.run(source, output, 17)
                self.assertFalse(output.exists())
        expected = {"scores": [{"value": 0.125}, {"value": -2.5}]}
        records[0] = dict(records[0], metadata=expected)
        source = self.write_records("split-finite.jsonl", records)
        output = self.work / "split-finite"
        SPLIT.run(source, output, 17)
        actual = {
            row["id"]: row
            for split in ("train", "validation", "test")
            for row in read_jsonl(output / f"{split}.jsonl")
        }
        self.assertEqual(actual[records[0]["id"]]["metadata"], expected)

    def test_cleaning_unicode_separator_stays_inside_record(self):
        text = "Before\u2028after the rain."
        source = self.write_records("separator.jsonl", [
            {"id": "separator", "source": "test", "language": "en", "kind": "prose", "text": text}
        ])
        args = self.clean_args(source, "clean-separator")
        report = CLEANING.run(args)
        self.assertEqual((report["input_records"], report["kept_records"]), (1, 1))
        self.assertEqual(read_jsonl(args.output / "kept.jsonl")[0]["text"], text)

    def test_cleaning_invalid_utf8_has_no_success_artifact(self):
        source = self.work / "invalid-utf8.jsonl"
        source.write_bytes(b'{"id":"broken","text":"\xff"}\n')
        args = self.clean_args(source, "clean-invalid-utf8")
        with self.assertRaises(UnicodeDecodeError):
            CLEANING.run(args)
        self.assertFalse(args.output.exists())

    def test_cleaning_surrogate_is_rejected_but_auditable(self):
        source = self.work / "surrogate.jsonl"
        record = {"id": "surrogate", "source": "test", "language": "en", "kind": "prose",
                  "text": "A broken scalar " + chr(0xD800)}
        source.write_text(json.dumps(record, ensure_ascii=True) + "\n", encoding="utf-8")
        args = self.clean_args(source, "clean-surrogate")
        report = CLEANING.run(args)
        self.assertEqual((report["kept_records"], report["rejected_records"]), (0, 1))
        rejected = read_jsonl(args.output / "rejected.jsonl")
        self.assertEqual(rejected[0]["primary_reason"], "invalid_unicode")
        self.assertEqual(rejected[0]["original_record"]["text"], record["text"])
        # 拒绝日志自身仍必须是可以严格解码的 UTF-8。
        (args.output / "rejected.jsonl").read_bytes().decode("utf-8")

    def test_cleaning_nonfinite_metadata_numbers_are_invalid_json(self):
        base = {"id": "number", "source": "test", "language": "en", "kind": "prose",
                "text": "This text is otherwise acceptable."}
        for index, literal in enumerate(("NaN", "Infinity", "-Infinity", "1e999", "-1e999")):
            with self.subTest(literal=literal):
                source = self.work / f"nonfinite-{index}.jsonl"
                payload = json.dumps(base)[:-1] + ', "weight": ' + literal + "}\n"
                source.write_text(payload, encoding="utf-8")
                args = self.clean_args(source, f"clean-nonfinite-{index}")
                report = CLEANING.run(args)
                self.assertEqual((report["kept_records"], report["rejected_records"]), (0, 1))
                self.assertEqual(report["primary_counts"]["invalid_json"], 1)
                rejected = read_jsonl(args.output / "rejected.jsonl")
                self.assertEqual(rejected[0]["primary_reason"], "invalid_json")
                self.assertEqual(rejected[0]["raw_line"], payload.rstrip("\n"))

    def test_cleaning_nested_overflow_and_finite_metadata(self):
        base = {"id": "nested", "source": "test", "language": "en", "kind": "prose",
                "text": "This text is otherwise acceptable."}
        for index, literal in enumerate(("1e999", "-1e999")):
            with self.subTest(literal=literal):
                source = self.work / f"nested-{index}.jsonl"
                payload = json.dumps(base)[:-1] + ', "metadata": {"scores": [{"value": ' + literal + "}]}}\n"
                source.write_text(payload, encoding="utf-8")
                args = self.clean_args(source, f"clean-nested-{index}")
                report = CLEANING.run(args)
                self.assertEqual((report["kept_records"], report["rejected_records"]), (0, 1))
                self.assertEqual(report["primary_counts"]["invalid_json"], 1)
        finite = dict(base, metadata={"scores": [{"value": 0.125}, {"value": -2.5}], "scale": 1000.0})
        source = self.write_records("finite.jsonl", [finite])
        args = self.clean_args(source, "clean-finite")
        report = CLEANING.run(args)
        self.assertEqual((report["kept_records"], report["rejected_records"]), (1, 0))
        self.assertEqual(read_jsonl(args.output / "kept.jsonl")[0]["metadata"], finite["metadata"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
