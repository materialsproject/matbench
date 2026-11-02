"""Entry point to the AlchemBERT source and reproducibility documentation.

Original source: https://gitee.com/liuxiaotong15/alchemBERT
The original submission offers trained per-task/per-fold models (about 25 GB)
on request. They were not present in the audited public source snapshot.
See README.md for recovered source, provenance and remaining evidence gaps.
"""

from pathlib import Path

if __name__ == "__main__":
    print(Path(__file__).with_name("README.md").read_text(encoding="utf-8"))
