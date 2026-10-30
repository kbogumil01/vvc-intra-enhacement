#!/usr/bin/env python3
"""
Generates a comparison table of BD-Rate results.
Anchor: VTM without filters (nofilter).

All models use the same anchor, so results are directly comparable.
"""
import csv
import os
import sys


results = {
    "VTM z filtrami (sam VTM)": {"Y": -3.34, "U": -12.25, "V": -12.58},
    "Nofilter + sieć 4ch": {"Y": -6.50, "U": -16.63, "V": -18.38},
    "Nofilter + sieć 6ch (boundary)": {"Y": -6.58, "U": -16.23, "V": -18.25},
    "Nofilter + sieć 8ch (all part.)": {"Y": -6.76, "U": -16.28, "V": -17.74},
    "Filter + sieć 6ch (binary bnd.)": {"Y": -5.69, "U": -22.89, "V": -24.38},
    "Filter + sieć 8ch (all part.)": {"Y": -5.65, "U": -22.99, "V": -24.52},
    "DB-only + sieć 4ch": {"Y": -6.16, "U": -15.95, "V": -17.48},
}


temporal_csv = os.path.join(os.path.dirname(__file__), "bd_results_temporal_ctc.csv")
if os.path.exists(temporal_csv):
    with open(temporal_csv) as f:
        rows = list(csv.DictReader(f))
    avg = [r for r in rows if r["sequence"] == "AVERAGE"]
    if avg:
        avg = avg[0]
        results["Nofilter + sieć temporal 12ch"] = {
            "Y": float(avg["bdr_psnr_y"]),
            "U": float(avg["bdr_psnr_u"]),
            "V": float(avg["bdr_psnr_v"]),
        }
    else:
        print("WARN: No AVERAGE row in temporal results", file=sys.stderr)
else:
    print(f"WARN: {temporal_csv} not found yet (job still running?)", file=sys.stderr)
    results["Nofilter + sieć temporal 12ch"] = {"Y": "???", "U": "???", "V": "???"}


print()
print("=" * 75)
print("BD-Rate [%] na CTC — anchor: VTM bez filtrów (nofilter)")
print("=" * 75)
print(f"{'Konfiguracja':<40} {'BD-Rate Y':>10} {'BD-Rate U':>10} {'BD-Rate V':>10}")
print("-" * 75)

for name, vals in results.items():
    y = f"{vals['Y']:.2f}%" if isinstance(vals['Y'], float) else vals['Y']
    u = f"{vals['U']:.2f}%" if isinstance(vals['U'], float) else vals['U']
    v = f"{vals['V']:.2f}%" if isinstance(vals['V'], float) else vals['V']
    print(f"{name:<40} {y:>10} {u:>10} {v:>10}")

print("-" * 75)
print()
print("Ujemne wartości = oszczędność bitrate przy tej samej jakości.")
print("Im bardziej ujemne, tym lepiej.")
print()


if os.path.exists(temporal_csv):
    print()
    print("=" * 75)
    print("Temporal 12ch — wyniki per sekwencja")
    print("=" * 75)
    print(f"{'Sekwencja':<40} {'BD-Rate Y':>10} {'BD-Rate U':>10} {'BD-Rate V':>10}")
    print("-" * 75)
    with open(temporal_csv) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        name = r["sequence"]
        y = float(r["bdr_psnr_y"])
        u = float(r["bdr_psnr_u"])
        v = float(r["bdr_psnr_v"])
        marker = " <<<" if name == "AVERAGE" else ""
        print(f"{name:<40} {y:>9.2f}% {u:>9.2f}% {v:>9.2f}%{marker}")
    print("-" * 75)


if __name__ == "__main__":
    pass
