#!/usr/bin/env python3
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table4_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--final_score", type=float, required=True)
    ap.add_argument("--final_psnr", type=float, required=True)
    ap.add_argument("--final_ssim", type=float, required=True)
    ap.add_argument("--final_lpips", type=float, required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    final_tex = rf"""\begin{{table*}}[t]
\centering
\small
\caption{{Final Codabench/test result of the selected single-checkpoint submission.}}
\label{{tab:final_score_filled}}
\begin{{tabular}}{{lccccp{{0.28\linewidth}}}}
\toprule
\textbf{{Submission}} & \textbf{{PSNR(Y)}} & \textbf{{SSIM(Y)}} & \textbf{{LPIPS}} & \textbf{{Final score}} & \textbf{{Notes}} \\
\midrule
Clean-310 + MoCE-Prompt Refiner v2-150k & {args.final_psnr:.2f} & {args.final_ssim:.2f} & {args.final_lpips:.2f} & {args.final_score:.2f} & Final single-checkpoint submission. \\
\bottomrule
\end{{tabular}}
\end{{table*}}
"""
    (out / "final_score_latex.tex").write_text(final_tex, encoding="utf-8")

    table4_tex = out / "table4_latex.tex"
    combined = []
    if table4_tex.exists():
        combined.append("% --- Table 4 internal validation metrics ---")
        combined.append(table4_tex.read_text(encoding="utf-8"))
    combined.append("% --- Final Codabench score table ---")
    combined.append(final_tex)
    (out / "report_ready_tables.tex").write_text("\n\n".join(combined), encoding="utf-8")

    print(f"[OK] final score LaTeX: {out/'final_score_latex.tex'}")
    print(f"[OK] combined tables: {out/'report_ready_tables.tex'}")

if __name__ == "__main__":
    main()
