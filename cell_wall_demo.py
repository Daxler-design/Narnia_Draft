import argparse
from pathlib import Path

import core


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a synthetic single-slice cell-wall field demo")
    ap.add_argument("--out", type=str, default="output/cell_wall_demo.npz")
    ap.add_argument("--nx", type=int, default=256)
    ap.add_argument("--ny", type=int, default=256)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--tau", type=float, default=12.0)
    ap.add_argument("--method", type=str, default="entropy", choices=["entropy", "top2gap"])
    ap.add_argument("--smooth", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--thickness", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    core.demo_cell_wall_single_slice_npz(
        str(out_path),
        nx=args.nx,
        ny=args.ny,
        k=args.k,
        tau=args.tau,
        wall_method=args.method,
        smooth_sigma=args.smooth,
        threshold=args.threshold,
        thickness_px=args.thickness,
        seed=args.seed,
    )

    print(f"W/B demo saved: {out_path}")


if __name__ == "__main__":
    main()
