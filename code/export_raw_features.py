"""Export the frozen raw representation without models, processors or labels."""
import argparse
import csv
from pathlib import Path
import numpy as np
from c05_reproduction import calculate_smiles, read_json, PKG, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.suffix == ".npz", "Output must have a .npz suffix")
    require(not args.output.exists(), "Output already exists")
    with args.input.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ["row_id", "Drug"], "Expected row_id,Drug columns")
        rows = list(reader)
    require(rows and all(r["row_id"] and r["Drug"] is not None for r in rows), "Empty input or missing cells")
    ids = [r["row_id"] for r in rows]
    bits, descriptors = calculate_smiles(ids, [r["Drug"] for r in rows])
    names = read_json(PKG / "manifests/c03/feature_schema.json")["rdkit2d"]["feature_names"]
    # Exclusive creation preserves an existing output even if another writer races.
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, row_id=np.asarray(ids, dtype=str), morgan=bits,
                            rdkit2d=descriptors, descriptor_names=np.asarray(names, dtype=str))
    print(f"Exported {len(ids)} rows; raw features only, no predictions.")


if __name__ == "__main__":
    main()
