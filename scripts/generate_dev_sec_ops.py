from __future__ import annotations

import argparse
import datetime as dt
import subprocess
from pathlib import Path
from xml.etree import ElementTree

import yaml


def _read_last_commit_hashes(limit: int = 5) -> list[str]:
    result = subprocess.run(
        ["git", "log", f"-n{limit}", "--pretty=format:%H"],
        capture_output=True,
        text=True,
        check=False,
    )
    hashes = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    while len(hashes) < limit:
        hashes.append("N/A")
    return hashes


def _read_coverage_percent(coverage_xml_path: Path | None) -> float | None:
    if coverage_xml_path is None or not coverage_xml_path.exists():
        return None
    root = ElementTree.parse(coverage_xml_path).getroot()
    line_rate = float(root.attrib.get("line-rate", "0"))
    return round(line_rate * 100, 2)


def build_manifest(
    image: str,
    digest: str,
    coverage_percent: float | None,
) -> dict:
    return {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "docker_image": {
            "name": image or "docker.io/<dockerhub_username>/bbc-news-classifier:latest",
            "digest": digest or "sha256:pending",
            "signature": {
                "tool": "cosign",
                "status": "pending_if_not_signed",
            },
        },
        "repository": {
            "last_5_commits": _read_last_commit_hashes(limit=5),
        },
        "quality": {
            "test_coverage_percent": coverage_percent,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dev_sec_ops.yml manifest.")
    parser.add_argument("--output", default="dev_sec_ops.yml", help="Path to output YAML file.")
    parser.add_argument("--image", default="", help="Docker image reference.")
    parser.add_argument("--digest", default="", help="Docker image digest.")
    parser.add_argument(
        "--coverage",
        default="coverage.xml",
        help="Path to coverage XML file (optional).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = Path(args.output)
    coverage_percent = _read_coverage_percent(Path(args.coverage))
    manifest = build_manifest(args.image, args.digest, coverage_percent)
    output_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    print(f"Manifest was written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
