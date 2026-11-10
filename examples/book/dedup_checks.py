"""去重教学例的行为检查；在指定远端 CPU 节点执行。"""

import itertools
import json
import unittest
from fractions import Fraction

import dedup_walkthrough as walkthrough


class DedupChecks(unittest.TestCase):
    def setUp(self):
        self.records = [json.loads(line) for line in walkthrough.FIXTURE.read_text(encoding="utf-8").splitlines()]

    def test_comparison_preserves_indentation(self):
        self.assertEqual(walkthrough.comparison_text("  cafe\u0301\r\n"), "  café\n")
        self.assertNotEqual(walkthrough.comparison_text(" x\n"), walkthrough.comparison_text("x\n"))

    def test_digest_collision_does_not_merge_different_text(self):
        unique, representatives, _ = walkthrough.exact_deduplicate(self.records, digest=lambda _: "collision")
        self.assertEqual([record["id"] for record in unique], ["d01", "d03", "d04", "d05"])
        self.assertEqual(representatives["d02"], "d01")

    def test_input_order_does_not_change_representatives(self):
        forward = walkthrough.exact_deduplicate(self.records)
        backward = walkthrough.exact_deduplicate(list(reversed(self.records)))
        self.assertEqual(forward, backward)

    def test_candidates_are_not_deletion_decisions(self):
        report, kept = walkthrough.build_report(self.records)
        self.assertEqual(report["signatures"], {
            "d01": ["a", "b", "c", "c"], "d03": ["a", "b", "d", "c"],
            "d04": ["b", "b", "d", "c"], "d05": ["a", "b", "b", "a"],
        })
        reviews = report["candidate_pair_reviews"]
        self.assertEqual([row["document_ids"] for row in reviews], [
            ["d01", "d03"], ["d01", "d05"], ["d03", "d04"], ["d03", "d05"],
        ])
        self.assertEqual([row["jaccard_fraction"] for row in reviews], ["3/4", "2/5", "3/4", "1/3"])
        self.assertEqual([record["id"] for record in kept], ["d01", "d04", "d05"])
        self.assertEqual(report["final_representatives"], {
            "d01": "d01", "d02": "d01", "d03": "d01", "d04": "d04", "d05": "d05",
        })
        counts = report["counts"]
        self.assertEqual(counts["input_documents"], counts["kept_documents"] + counts["exact_removed_documents"] + counts["near_removed_documents"])
        self.assertEqual(counts["candidate_pairs_reviewed"], 4)
        self.assertEqual(counts["pairs_at_or_above_threshold"], 2)
        for row in report["exhaustive_permutation_checks"]:
            self.assertEqual(row["collision_fraction"], row["jaccard_fraction"])
            self.assertEqual(row["permutations"], 720)

    def test_unseen_candidate_can_remain_even_when_similar(self):
        sets = {"a": set("abc"), "b": set("abcd")}
        kept, _, _ = walkthrough.select_representatives(sets, [], Fraction(3, 4))
        self.assertEqual(kept, ["a", "b"])

    def test_small_universe_collision_identity(self):
        universe = tuple("abc")
        nonempty = [set(items) for size in range(1, 4) for items in itertools.combinations(universe, size)]
        for left, right in itertools.product(nonempty, repeat=2):
            matches = sum(
                walkthrough.minhash_signature(left, (order,)) == walkthrough.minhash_signature(right, (order,))
                for order in itertools.permutations(universe)
            )
            self.assertEqual(Fraction(matches, 6), walkthrough.jaccard(left, right))

    def test_empty_or_malformed_signature_is_rejected(self):
        with self.assertRaises(ValueError):
            walkthrough.minhash_signature(set(), walkthrough.PERMUTATIONS)
        with self.assertRaises(ValueError):
            walkthrough.minhash_signature({"a"}, (tuple("aab"),))
        with self.assertRaises(ValueError):
            walkthrough.lsh_candidates({"a": ("a", "b", "c")}, 2)
        with self.assertRaises(ValueError):
            walkthrough.jaccard(set(), set())


if __name__ == "__main__":
    unittest.main()
