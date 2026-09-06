from __future__ import annotations

import os
import re
from pathlib import Path


def main() -> None:
    version = os.environ["TAGPR_NEXT_VERSION"].removeprefix("v")
    citation_path = Path("CITATION.cff")
    citation = citation_path.read_text()
    updated, count = re.subn(
        r"^version: .+$",
        f"version: {version}",
        citation,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError(f"Expected one software version in {citation_path}, found {count}")
    citation_path.write_text(updated)


if __name__ == "__main__":
    main()
