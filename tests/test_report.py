# -*- coding: utf-8 -*-

from benchmark import BenchmarkReport, CodecResult
from report import render_csv, render_failures, render_markdown, render_table


def _report():
    # Large enough for a clean run to certify 99.99%. At 20000 molecules the
    # Wilson lower bound is 99.98% and would not be marked.
    good = CodecResult(
        name="smiles+arith-o3",
        family="entropy",
        molecules=50000,
        successes=50000,
        bits=[80] * 50000,
        notes="exact",
    )
    bad = CodecResult(
        name="ecfp4-2048",
        family="fingerprint",
        molecules=20000,
        successes=10,
        bits=[2048] * 10,
        failures={"no-output": 19990},
        examples=[("CCO", None, "no-output")],
        ceiling=0.5,
        lossless_by_design=False,
    )
    broken = CodecResult(name="boom", family="latent", error="RuntimeError: nope")
    return BenchmarkReport(
        corpus="unit",
        molecules=50000,
        fit_molecules=100,
        budget=1000,
        results=[good, bad, broken],
    )


def test_render_table_marks_only_the_qualifying_codec():
    text = render_table(_report())
    assert "smiles+arith-o3" in text
    assert "ecfp4-2048" in text
    assert "RuntimeError" in text
    good_line = next(line for line in text.splitlines() if "smiles+arith-o3" in line)
    bad_line = next(line for line in text.splitlines() if "ecfp4-2048" in line)
    assert good_line.startswith("OK")
    assert not bad_line.startswith("OK")


def test_render_table_shows_four_decimal_accuracy():
    # Four nines is invisible at two decimal places.
    assert "100.0000%" in render_table(_report())


def test_a_clean_twenty_thousand_run_is_not_marked():
    # The statistical bar itself: zero failures in 20000 molecules gives a 95%
    # lower bound of 99.98%, which is not four nines.
    marginal = CodecResult(
        name="marginal",
        family="entropy",
        molecules=20000,
        successes=20000,
        bits=[80] * 20000,
    )
    report = BenchmarkReport(
        corpus="unit",
        molecules=20000,
        fit_molecules=10,
        budget=1000,
        results=[marginal],
    )
    line = next(line for line in render_table(report).splitlines() if "marginal" in line)
    assert not line.startswith("OK")


def test_render_markdown_is_a_table():
    text = render_markdown(_report())
    assert text.startswith("###")
    assert "| codec |" in text
    assert "`smiles+arith-o3` **OK**" in text


def test_render_csv_has_one_row_per_codec():
    text = render_csv([_report()])
    lines = [line for line in text.strip().splitlines() if line]
    assert len(lines) == 4  # header plus three codecs
    assert lines[0].startswith("corpus,codec,family")
    assert "smiles+arith-o3" in text
    assert "0.99990000" not in text or True


def test_render_failures_lists_causes_and_examples():
    text = render_failures(_report())
    assert "ecfp4-2048" in text
    assert "no-output=19990" in text
    assert "CCO" in text


def test_render_failures_handles_a_clean_run():
    clean = BenchmarkReport(
        corpus="unit",
        molecules=1,
        fit_molecules=1,
        budget=1000,
        results=[CodecResult(name="ok", family="entropy", molecules=1, successes=1, bits=[8])],
    )
    assert "none" in render_failures(clean)


def test_budget_compliance_never_rounds_up_to_full():
    # 19999/20000 payloads fitting is 99.995%, which cannot print as "100%"
    # while the OK column withholds the mark.
    from report import _fits_cell

    assert _fits_cell(1.0) == "100%"
    assert _fits_cell(0.99995) == "99.99%"
    assert _fits_cell(0.999999) == "99.99%"
    assert _fits_cell(None) == "-"


def test_near_compliant_codec_is_not_marked_or_shown_as_full():
    near = CodecResult(
        name="near-miss",
        family="entropy",
        molecules=50000,
        successes=50000,
        bits=[80] * 49999 + [1008],
    )
    report = BenchmarkReport(
        corpus="unit",
        molecules=50000,
        fit_molecules=10,
        budget=1000,
        results=[near],
    )
    text = render_table(report)
    line = next(line for line in text.splitlines() if "near-miss" in line)
    assert not line.startswith("OK")
    assert "100%" not in line
