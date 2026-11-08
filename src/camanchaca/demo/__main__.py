"""`python -m camanchaca.demo --exp main [--share]` (equivale a `python app/app.py`)."""
import sys

from .app import main

if __name__ == "__main__":
    sys.exit(main())
