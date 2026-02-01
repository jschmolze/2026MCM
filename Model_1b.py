import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# -----------------------------
# Constants
# -----------------------------
S = 365.25 * 24 * 3600  # seconds per year
g0 = 9.80665  # gravitational acceleration (m/s^2)
D = 3.84e8  # distance (m)
M_total = 100_000_000.0  # total mass (kg)
m_f = 150.0  # mass per flight (kg)
N = M_total / m_f  # number of flights

# Launch capacity parameters
P_E = 10  # number of launch pads
L_pad = 365  # launches per pad per year
T_min = N / (P_E * L_pad)  # minimum feasible timeline

# Rocket parameters
k = 2.0  # velocity multiplier for time-driven increment
Isp = 450.0  # specific impulse (s) - typical for RP-1/LOX
v_e = Isp * g0  # exhaust velocity (m/s)
c_p = 2_000_000.0  # cost per kg propellant (USD/kg)

# Delta-v components
delta_v_fixed = 9400.0  # fixed delta-v (m/s) - typical for LEO
t_burn = 150.0  # burn time (s) - typical for first stage
delta_v_grav = g0 * t_burn  # gravity losses (m/s)

# Drag parameter
alpha = 1e-6  # drag coefficient (s^2/m^2) - adjust as needed

def delta_v_total(T):
    """Calculate total delta-v requirement as function of timeline T (years)"""
    T = np.asarray(T, dtype=float)
    v_T = (D * N) / (T * S)  # average speed per trip (m/s)
    
    delta_v_drag = alpha * v_T**2
    delta_v_time = k * v_T
    
    dv_total = delta_v_fixed + delta_v_grav + delta_v_drag + delta_v_time
    return dv_total

def total_cost(T):
    """Calculate total propellant cost as function of timeline T (years)"""
    T = np.asarray(T, dtype=float)
    T_masked = np.where(T >= T_min, T, np.nan)
    
    dv_total = delta_v_total(T_masked)
    exponent = dv_total / v_e
    
    # Total cost
    return c_p * N * m_f * (np.exp(exponent) - 1.0)

# -----------------------------
# Visualization
# -----------------------------
T_grid = np.linspace(1, 2500, 2500)
C_grid = total_cost(T_grid)

plt.figure(figsize=(12, 7))
ax = plt.gca()

# Plot valid curve
plt.plot(T_grid, C_grid, label='Total Propellant Cost $C(T)$', color='teal', linewidth=3)

# Highlight Constraint
plt.axvspan(0, T_min, color='red', alpha=0.1, label='Capacity Constraint (Impossible)')
plt.axvline(T_min, color='red', linestyle='--', alpha=0.5)

# X-AXIS SCALE
plt.xlim(0, 2500)
plt.xticks(np.arange(0, 2501, 250))

# Y-AXIS SCALE
ax.set_yscale('log')
plt.ylim(1e13, 1e20)  # Adjusted for potentially higher costs with drag
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20))
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=20))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'$10^{{{int(np.log10(y))}}}$'))

# Labels and Styling
plt.title('Rocket Cost vs Timeline (with Gravity & Drag Losses)', fontsize=14)
plt.xlabel('Total Project Duration $T$ (Years)', fontsize=12)
plt.ylabel('Total Propellant Cost (USD)', fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.2)

# Dynamic Annotation
plt.text(T_min + 50, ax.get_ylim()[0] * 5, 
         rf"Min $T \approx {T_min:.1f}$ yrs", 
         color='red', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# Print costs at key timelines
print("=" * 60)
print("ROCKET COST ANALYSIS (Earth Launch with Gravity & Drag)")
print("=" * 60)
print(f"Number of launch pads (P_E): {P_E}")
print(f"Launches per pad per year: {L_pad}")
print(f"Minimum feasible timeline: {T_min:.2f} years")
print(f"Fixed delta-v: {delta_v_fixed:,.0f} m/s")
print(f"Gravity losses: {delta_v_grav:.2f} m/s")
print()

for T_val in [T_min, 558.7, 1000, 2000, 5000]:
    if T_val >= T_min:
        cost = total_cost(T_val)
        dv = delta_v_total(T_val)
        print(f"T = {T_val:,.1f} years:")
        print(f"  Total Δv: {dv:,.2f} m/s")
        print(f"  Total cost: ${cost:.2e}")
        print()
print("=" * 60)