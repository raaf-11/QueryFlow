"""
Download the Chinook sample database and verify it.

    python setup_db.py

Chinook is a small fake digital music store: artists, albums, tracks,
customers, invoices. It is the standard demo database for Text-to-SQL work
because it has enough tables and foreign keys to make joins non-trivial.
"""

from __future__ import annotations

import sqlite3
import sys
import urllib.request
from pathlib import Path

DEST = Path("chinook.db")

# Tried in order. The GitHub release asset is the canonical one.
SOURCES = [
    "https://github.com/lerocha/chinook-database/releases/download/v1.4.5/Chinook_Sqlite.sqlite",
    "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite",
]

EXPECTED_TABLES = {
    "Album", "Artist", "Customer", "Employee", "Genre", "Invoice",
    "InvoiceLine", "MediaType", "Playlist", "PlaylistTrack", "Track",
}


def download() -> bool:
    for url in SOURCES:
        try:
            print(f"Downloading from {url} ...")
            urllib.request.urlretrieve(url, DEST)
            if DEST.stat().st_size > 500_000:
                return True
            print("  file looked too small, trying the next source")
        except Exception as e:
            print(f"  failed: {e}")
    return False


def verify() -> bool:
    con = sqlite3.connect(DEST)
    try:
        found = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        missing = EXPECTED_TABLES - found
        if missing:
            print(f"Missing expected tables: {sorted(missing)}")
            return False

        print(f"\nDatabase ready: {DEST.resolve()} ({DEST.stat().st_size // 1024} KB)")
        print(f"{len(found)} tables\n")
        for t in sorted(EXPECTED_TABLES):
            n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            print(f"  {t:<16} {n:>6} rows")
        return True
    finally:
        con.close()


if __name__ == "__main__":
    if DEST.exists():
        print(f"{DEST} already exists. Delete it first to re-download.\n")
    elif not download():
        print(
            "\nCould not download automatically.\n"
            "Download Chinook_Sqlite.sqlite manually from\n"
            "  https://github.com/lerocha/chinook-database/releases\n"
            f"and save it as {DEST.resolve()}"
        )
        sys.exit(1)

    sys.exit(0 if verify() else 1)
