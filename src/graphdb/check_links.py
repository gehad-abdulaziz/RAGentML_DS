"""Check that every resource URL in data/graph_seed/resources.csv really opens.

Owner: Member 4 (feature/graph-tool).
Run:  python -m src.graphdb.check_links
(data_sources.md requires resource links to be real and checked.)

Result meanings:
  OK        the page opened (HTTP 200)
  BLOCKED   the site answered but refuses scripts (403/429). The page exists: open it in a browser.
  UNREACHABLE  this computer could not connect (network/SSL). Open the link in a browser to confirm.
"""
import csv
import ssl
import urllib.error
import urllib.request

from src import config

HEADERS = {"User-Agent": "Mozilla/5.0 (link checker for a university project)"}


def fetch_status(url: str) -> str:
    """Return 'OK', 'BLOCKED' or 'UNREACHABLE: <reason>' for one URL."""
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return "OK" if resp.status == 200 else f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        return "BLOCKED" if exc.code in (403, 429) else f"HTTP {exc.code}"
    except Exception as exc:
        # Some Windows Pythons lack the certificate chain (CERTIFICATE_VERIFY_FAILED).
        # We only READ a public page here, so retry once without certificate checking.
        if "CERTIFICATE_VERIFY_FAILED" in str(exc):
            try:
                ctx = ssl._create_unverified_context()
                with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
                    return "OK" if resp.status == 200 else f"HTTP {resp.status}"
            except urllib.error.HTTPError as exc2:
                return "BLOCKED" if exc2.code in (403, 429) else f"HTTP {exc2.code}"
            except Exception as exc2:
                return f"UNREACHABLE: {type(exc2).__name__}"
        return f"UNREACHABLE: {type(exc).__name__}"


def main() -> None:
    to_check = 0
    with open(config.GRAPH_SEED_DIR / "resources.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            status = fetch_status(row["url"])
            if status != "OK":
                to_check += 1
            print(f"{status:<28} {row['title']}  ->  {row['url']}")
    print(f"\n{to_check} link(s) to open manually in a browser." if to_check else "\nAll links work.")


if __name__ == "__main__":
    main()