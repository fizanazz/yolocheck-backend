"""
app/api/router.py

FIX 7: This file previously duplicated the router includes already in main.py.
        The api_router object was never actually used anywhere in the app.
        Keeping this file for reference only — main.py now handles all includes.
        Do NOT call app.include_router(api_router) — that would register
        every route twice, causing duplicate endpoint warnings and odd behaviour.
"""

# This file is intentionally left without active router includes.
# All route registration happens in app/main.py directly.
# See app/main.py for the authoritative list of included routers.
