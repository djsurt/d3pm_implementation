"""
Verify the downstream sequence-classifier augmentation study.

Run (needs the comparison py3.12 venv with scikit-learn):

    cd comparison
    PYTHONPATH=lib .venv/bin/python verify_seq_classifier.py

A lighter sweep than the full report (budgets 5/50/all, seeds 0,1) so it finishes
quickly while still exercising the whole pipeline and the headline claim.

EXPECTED OUTPUT (run_01 D3PM synthetic):
  * real-only F1 RISES monotonically with budget (more real data -> better).
  * real+synth F1 >= real-only at every scarce budget.
  * LIFT is clearly positive at budget=5 (~+0.03..+0.05), and ~0 at budget='all'
    (the task saturates, so synthetic can no longer help).
  * lift(budget=5) > lift(budget='all').
All four CHECK lines should print PASS.
"""

from pathlib import Path

from seq_classifier import budget_sweep, print_report

HERE = Path(__file__).parent


def main() -> None:
    report = budget_sweep(
        malicia_dir=HERE / ".." / "malicia",
        run_dir=HERE / ".." / "runs" / "run_01_baseline",
        families=["winwebsec", "zbot", "zeroaccess"],
        budgets=[5, 50, None],  # None == "all"
        seeds=(0, 1),
    )
    print_report(report)

    sweep = {("all" if r["budget"] is None else r["budget"]): r for r in report["sweep"]}
    lift_small = sweep[5]["lift"]
    lift_all = sweep["all"]["lift"]
    ro = [r["real_only_f1"] for r in report["sweep"]]

    checks = [
        ("real-only F1 rises with budget", ro == sorted(ro)),
        ("synth helps at scarce budget (lift@5 > 0)", lift_small > 0),
        ("lift shrinks toward 0 at full data (lift@all < 0.01)", lift_all < 0.01),
        ("lift@5 > lift@all", lift_small > lift_all),
    ]
    print("Checks:")
    allpass = True
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        allpass = allpass and ok
    print(f"\n{'ALL CHECKS PASS' if allpass else 'SOME CHECKS FAILED'}")


if __name__ == "__main__":
    main()
