"""Entry point: python -m vmo_pipecat"""

import asyncio
from .runtime import main

if __name__ == "__main__":
    asyncio.run(main())
