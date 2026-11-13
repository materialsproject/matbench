"""Material-identity checks must reject wrong formulas, amounts and targets."""

import unittest

from verify_inputs import alloy_mismatches, formula_from_text, output_matches


class InputAuditTests(unittest.TestCase):
    def test_formula_identity_is_exact(self):
        text = "The chemical formula Ga1As0.1P0.9 consists of: 1.0 atoms of Ga"
        self.assertEqual(formula_from_text(text), "Ga1As0.1P0.9")
        self.assertNotEqual(formula_from_text(text), "G1128Ga1As0.1P0.9")
        with self.assertRaises(ValueError):
            formula_from_text("A different unrecognized template")

    def test_alloy_uses_displayed_rounding_without_ignoring_elements(self):
        text = (
            "This is an alloy with the following composition: 62.5% iron (Fe), "
            "9.4% chromium (Cr), and smaller amounts of niobium (Nb) (0.0060%)."
        )
        self.assertEqual(alloy_mismatches(text, "Fe0.625Cr0.0936Nb0.0000604"), [])
        self.assertTrue(alloy_mismatches(text, "Fe0.625Cr0.0946Nb0.0000604"))
        self.assertTrue(alloy_mismatches(text, "Fe0.625Cr0.0936Nb0.0000604C0.01"))
        with self.assertRaises(ValueError):
            alloy_mismatches(text + " 1.0% iron (Fe)", "Fe0.625Cr0.0936Nb0.0000604")

    def test_output_equality_rejects_wrong_type_nan_and_rounded_labels(self):
        self.assertTrue(output_matches("False", False))
        self.assertFalse(output_matches("0", False))
        self.assertTrue(output_matches("1.22", 1.22))
        self.assertFalse(output_matches("1.2", 1.22))
        self.assertFalse(output_matches("nan", 1.22))


if __name__ == "__main__":
    unittest.main()
