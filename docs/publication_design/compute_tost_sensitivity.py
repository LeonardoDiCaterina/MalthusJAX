from scipy.stats import norm, t, chi2
from scipy.integrate import quad
import csv


def power(n, r, mu):
    df = n - 1
    critical = t.ppf(0.975, df)
    sn = n**0.5

    def integrand(u):
        a = sn / r - critical * (chi2.ppf(u, df) / df) ** 0.5
        if a <= 0:
            return 0.0
        shift = mu * sn / r
        return norm.cdf(a - shift) - norm.cdf(-a - shift)

    return quad(integrand, 0, 1, epsabs=1e-7, limit=100)[0]


rows = []
for r in [0.5, 1, 2, 4, 8]:
    for n in [30, 50, 100, 200, 500]:
        rows.append(
            {
                "independent_pairs": n,
                "paired_sd_over_margin": r,
                "power_true_difference_zero": round(power(n, r, 0), 5),
                "power_true_difference_half_margin": round(power(n, r, 0.5), 5),
                "per_endpoint_alpha": 0.025,
                "method": "normal paired differences; exact t-TOST integral over chi-square sample variance",
            }
        )
with open("docs/publication_design/tost_sensitivity.csv", "w") as f:
    w = csv.DictWriter(f, fieldnames=rows[0])
    w.writeheader()
    w.writerows(rows)
for r in rows:
    if r["independent_pairs"] in (50, 200):
        print(r)
