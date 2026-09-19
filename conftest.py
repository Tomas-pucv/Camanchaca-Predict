# Hace visible el paquete src/camanchaca al ejecutar pytest desde la raíz.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
