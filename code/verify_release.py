"""Verify release files without scientific dependencies or deserialization."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "release_manifest.json").read_text())
    for relative, expected in manifest["files_sha256"].items():
        path = root / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError("Missing or unsafe release file: " + relative)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Release hash mismatch: " + relative)
    inputs = root / "chemrxiv_submission/reproducibility/inputs"
    inventory = json.loads((inputs / "manifest.json").read_text())["sha256"]
    absent = [relative for relative in inventory if not (inputs / relative).is_file()]
    for directory in ("data", "generated"):
        if (root / directory).exists():
            print("Local runtime directory is outside the upload allowlist: " + directory)
    print(json.dumps({"release_files": "PASS", "verified_files": len(manifest["files_sha256"]),
                      "excluded_legacy_input_files": len(absent), "legacy_replay_inputs_complete": not absent,
                      "public_raw_rebuild_code_included": manifest.get("public_raw_rebuild_code_included", False),
                      "full_public_GPU_rebuild_verified": manifest.get("full_public_GPU_rebuild_verified", False),
                      "license_status": manifest["license_status"]}))


if __name__ == "__main__":
    main()
