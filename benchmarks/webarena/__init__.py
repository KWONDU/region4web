"""WebArena URL/port configuration for BrowserGym.

BrowserGym's ``WebArenaInstance`` reads site URLs from ``WA_*`` environment
variables (``WA_SHOPPING``, ``WA_GITLAB``, ..., ``WA_HOMEPAGE``) and then
copies them to the un-prefixed names (``SHOPPING``, ...) for the underlying
``libwebarena`` package. This module exports both sets before any gym.make
call, so multiple docker port layouts can be swapped via config.yaml.

Usage:
    from benchmarks.webarena import configure_webarena_env
    configure_webarena_env({
        "shopping": "http://host:17770",
        "shopping_admin": "http://host:17780/admin",
        ...
    })
    # now safe to import browsergym.webarena or gym.make("browsergym/webarena.N")
"""

from __future__ import annotations

import os
from typing import Dict, Optional

# site key → un-prefixed env var name; BrowserGym expects "WA_" + this on input.
_SITE_ENV_VARS = {
    "shopping": "SHOPPING",
    "shopping_admin": "SHOPPING_ADMIN",
    "reddit": "REDDIT",
    "gitlab": "GITLAB",
    "wikipedia": "WIKIPEDIA",
    "map": "MAP",
    "homepage": "HOMEPAGE",
}


def configure_webarena_env(urls: Dict[str, Optional[str]]) -> None:
    """Export WebArena site URLs as environment variables for BrowserGym.

    Sets both ``WA_<SITE>`` (which BrowserGym's WebArenaInstance asserts)
    and ``<SITE>`` (which libwebarena's env_config reads). Missing or null
    values are skipped.
    """
    for site, url in urls.items():
        if site not in _SITE_ENV_VARS:
            raise ValueError(f"Unknown WebArena site: {site}. Known: {list(_SITE_ENV_VARS)}")
        if url is None or url == "":
            continue
        base = _SITE_ENV_VARS[site]
        os.environ[f"WA_{base}"] = url
        os.environ[base] = url
