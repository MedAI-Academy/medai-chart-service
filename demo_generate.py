"""
Demo: Generate a publication-quality Kaplan-Meier curve
simulating a KEYNOTE-189-like trial (Pembrolizumab + Chemo vs Chemo)
"""
import sys
sys.path.insert(0, "/home/claude/medai-chart-service")

import numpy as np
from charts.kaplan_meier import render_kaplan_meier

# ── Simulate realistic NSCLC OS data ───────────────────────────
np.random.seed(42)

def simulate_arm(n, median_os, censor_rate=0.25, max_follow=36):
    """Simulate exponential survival data with random censoring."""
    lam = np.log(2) / median_os
    true_times = np.random.exponential(1 / lam, n)
    censor_times = np.random.uniform(0, max_follow, n)
    is_censored = np.random.random(n) < censor_rate
    
    times = []
    events = []
    for i in range(n):
        if is_censored[i]:
            t = min(censor_times[i], true_times[i])
            e = 0 if censor_times[i] < true_times[i] else 1
        else:
            t = min(true_times[i], max_follow)
            e = 1 if true_times[i] <= max_follow else 0
        times.append(round(t, 1))
        events.append(e)
    return times, events


# Experimental arm: median OS ~22 months
exp_times, exp_events = simulate_arm(200, median_os=22, censor_rate=0.30)
# Control arm: median OS ~12 months  
ctrl_times, ctrl_events = simulate_arm(200, median_os=12, censor_rate=0.25)


# ── Build request object (simulating the Pydantic model) ───────
class Arm:
    def __init__(self, label, times, events, color=None):
        self.label = label
        self.times = times
        self.events = events
        self.color = color

class Req:
    def __init__(self):
        self.arms = [
            Arm("Pembrolizumab + Chemo (n=200)", exp_times, exp_events),
            Arm("Placebo + Chemo (n=200)", ctrl_times, ctrl_events),
        ]
        self.title = "Overall Survival — Intent-to-Treat Population"
        self.xlabel = "Time (months)"
        self.ylabel = "Overall Survival (%)"
        self.show_ci = True
        self.show_censoring = True
        self.show_at_risk = True
        self.show_median = True
        self.hr_text = "HR 0.56 (95% CI 0.44–0.71)\nStratified log-rank p < 0.001"
        self.width = 10
        self.height = 7
        self.dpi = 200  # Lower for demo, 300 for production


# ── Render ──────────────────────────────────────────────────────
req = Req()
buf = render_kaplan_meier(req)

output_path = "/home/claude/medai-chart-service/demo_km_curve.png"
with open(output_path, "wb") as f:
    f.write(buf.getvalue())

print(f"✓ KM curve saved to {output_path}")
print(f"  Size: {len(buf.getvalue()) / 1024:.0f} KB")
