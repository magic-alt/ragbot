from __future__ import annotations

import sys
from pathlib import Path

# The SDK is an independently publishable src-layout package. The dedicated SDK
# workflow installs it normally; Ragbot's root pytest run intentionally installs
# only the main package, so expose this package's local `src/` for source tests.
SDK_SRC = Path(__file__).resolve().parents[1] / "src"
if str(SDK_SRC) not in sys.path:
    sys.path.insert(0, str(SDK_SRC))
