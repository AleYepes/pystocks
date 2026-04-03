from .ingest.session import *  # noqa: F401,F403

if __name__ == "__main__":
    import asyncio

    from .ingest.session import IBKRSession

    session = IBKRSession()
    asyncio.run(session.login())
