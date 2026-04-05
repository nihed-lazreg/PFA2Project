"""CLI entry-point for the banking signature verification system.

Usage examples:
    # Launch the Streamlit UI
    python main.py ui

    # Enroll a client
    python main.py enroll --client 42 --sigs data/real/original_42_*.png

    # Verify a signature against a specific client
    python main.py verify --client 42 --sig data/real/original_42_1.png

    # Identify an unknown signature across all clients
    python main.py identify --sig data/real/original_1_5.png
"""

import argparse
import glob
import logging
import os
import subprocess
import sys

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────────────────

def cmd_ui(_args):
    """Launch the Streamlit UI."""
    app_path = os.path.join(os.path.dirname(__file__), "interface", "app.py")
    logger.info("Starting Streamlit UI at %s", app_path)
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", app_path],
        check=True,
    )


def cmd_enroll(args):
    """Enroll a client from the command line."""
    from container import build_services
    from config.settings import Settings

    enrollment, *_ = build_services(Settings())

    # Expand glob patterns from shell-quoted arguments
    paths = []
    for pattern in args.sigs:
        expanded = glob.glob(pattern)
        paths.extend(expanded if expanded else [pattern])

    result = enrollment.enroll(
        client_id=args.client,
        signature_paths=paths,
        overwrite=args.overwrite,
    )

    if result.success:
        action = "updated" if result.is_update else "enrolled"
        print(
            f"✅ Client '{result.client_id}' {action}: "
            f"{result.num_enrolled} embeddings stored, "
            f"{result.num_skipped} skipped."
        )
    else:
        print(f"❌ Enrollment failed: {result.message}")
        sys.exit(1)


def cmd_verify(args):
    """Verify a signature against a specific client."""
    from container import build_services
    from config.settings import Settings

    _, verification, *_ = build_services(Settings())

    result = verification.verify(
        client_id=args.client,
        signature_path=args.sig,
        threshold=args.threshold,
    )

    if result.status == "ERROR":
        print(f"❌ Error: {result.message}")
        sys.exit(1)
    elif result.is_authentic:
        print(
            f"✅ AUTHENTIC – Client '{result.client_id}' | "
            f"Distance: {result.cosine_distance:.4f} | "
            f"Confidence: {result.confidence_pct:.1f}%"
        )
    else:
        print(
            f"🚨 REJECTED – Signature does not match client '{result.client_id}' | "
            f"Distance: {result.cosine_distance:.4f} | "
            f"Threshold: {result.threshold_used}"
        )
        sys.exit(2)


def cmd_identify(args):
    """Identify an unknown signature across all enrolled clients."""
    from container import build_services
    from config.settings import Settings

    _, _, identification, *_ = build_services(Settings())

    result = identification.identify(
        signature_path=args.sig,
        top_k=args.top_k,
        threshold=args.threshold,
    )

    if result.status == "ERROR":
        print(f"❌ Error: {result.message}")
        sys.exit(1)

    print(f"\nStatus: {result.status}")
    if result.best_match:
        print(
            f"Best match: Client '{result.best_match.client_id}' | "
            f"Distance: {result.best_match.cosine_distance:.4f} | "
            f"Confidence: {result.best_match.confidence_pct:.1f}%"
        )

    print("\nTop candidates:")
    for i, c in enumerate(result.top_candidates, 1):
        print(
            f"  {i:2d}. Client {c.client_id:6s} | "
            f"dist={c.cosine_distance:.4f} | "
            f"conf={c.confidence_pct:.1f}%"
        )

    if result.status == "UNKNOWN":
        sys.exit(2)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Banking signature verification system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ui
    sub.add_parser("ui", help="Launch the Streamlit web interface")

    # enroll
    p_enroll = sub.add_parser("enroll", help="Enroll a new client")
    p_enroll.add_argument("--client", required=True, help="Client ID")
    p_enroll.add_argument(
        "--sigs", nargs="+", required=True,
        help="Paths to reference signature images (globs accepted)",
    )
    p_enroll.add_argument(
        "--overwrite", action="store_true",
        help="Replace existing enrollment if client already exists",
    )

    # verify
    p_verify = sub.add_parser("verify", help="1:1 verification against a specific client")
    p_verify.add_argument("--client", required=True, help="Client ID")
    p_verify.add_argument("--sig", required=True, help="Path to query signature")
    p_verify.add_argument(
        "--threshold", type=float, default=None,
        help="Cosine-distance threshold (default: from settings)",
    )

    # identify
    p_identify = sub.add_parser("identify", help="1:N identification across all clients")
    p_identify.add_argument("--sig", required=True, help="Path to query signature")
    p_identify.add_argument("--top-k", type=int, default=5, help="Number of candidates")
    p_identify.add_argument(
        "--threshold", type=float, default=None,
        help="Cosine-distance threshold (default: from settings)",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "ui": cmd_ui,
        "enroll": cmd_enroll,
        "verify": cmd_verify,
        "identify": cmd_identify,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
