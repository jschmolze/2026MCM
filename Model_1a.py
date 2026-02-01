import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# -----------------------------
# Constants
# -----------------------------
S = 365.25 * 24 * 3600  # seconds per year
g0 = 9.80665  # (not used since we're in space)
D = 3.84e8  # distance (m)
M_total = 100_000_000.0  # total mass (kg)
m_f = 150.0  # mass per flight (kg)
N = M_total / m_f  # number of flights
R_max = 179_000.0  # max rate (kg/year)
T_min = M_total / R_max  # ~558.7 years
k = 2.0  # Changed from 2.0 to 1.0 (no gravity, so k=1 for one-way trip)
Isp = 900.0  # specific impulse (s)
v_e = Isp * g0  # exhaust velocity (m/s) - still use g0 as unit conversion
c_p = 2_000_000.0  # cost per kg propellant (USD/kg)

def total_cost(T):
    T = np.asarray(T, dtype=float)
    T_masked = np.where(T >= T_min, T, np.nan)
    # Simplified rocket equation: Δv = v_e * ln(mass_ratio)
    # So: mass_ratio = exp(Δv / v_e)
    # Δv = k * D / (T * S / N) = k * D * N / (T * S)
    exponent = (k * D * N) / (T_masked * S * v_e)
    return c_p * N * m_f * (np.exp(exponent) - 1.0)

# -----------------------------
# Visualization
# -----------------------------
# Extend grid to 2500 to push T_min to the left
T_grid = np.linspace(1, 2500, 2500)
C_grid = total_cost(T_grid)

plt.figure(figsize=(12, 7))
ax = plt.gca()

# Plot valid curve
plt.plot(T_grid, C_grid, label='Total Propellant Cost $C(T)$', color='teal', linewidth=3)

# Highlight Constraint
plt.axvspan(0, T_min, color='red', alpha=0.1, label='Capacity Constraint (Impossible)')
plt.axvline(T_min, color='red', linestyle='--', alpha=0.5)

# 1. FIX: X-AXIS SCALE
plt.xlim(0, 2500)
plt.xticks(np.arange(0, 2501, 250))

# 2. FIX: Y-AXIS GRANULARITY
ax.set_yscale('log')
plt.ylim(1e13, 1e17)  # Adjust these to encompass your data
# Show a tick for every power of 10
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20))
# Show minor ticks (2, 4, 6, 8) between powers of 10
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=20))
# Format labels as scientific notation ($10^x$)
ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())

# Force matplotlib to actually show all the ticks within the data range
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'$10^{{{int(np.log10(y))}}}$'))

# Labels and Styling
plt.title('Cost vs Length of Timeline', fontsize=14)
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

print(f"Total cost at 558.7 years: ${total_cost(558.7):.2e}")
print(f"Total cost at 1000 years: ${total_cost(1000):.2e}")
print(f"Total cost at 2000 years: ${total_cost(2000):.2e}")
print(f"Total cost at 5000 years: ${total_cost(5000):.2e}")

# -----------------------------
# Elevator Constants
# -----------------------------
m_e = 1000.0  # mass of elevator car (kg) - you need to specify this
m_p = 150.0  # mass of payload per trip (kg)
h = 100_000_000.0  # height in meters (100,000 km)
g = 9.81  # gravitational acceleration (m/s^2)
v_mach2 = 2450 * 1000 / 3600  # velocity in m/s (2450 km/h converted)
t_delivery = 1.0  # time to deliver energy uniformly (s)

M_total = 100_000_000.0  # total mass to transport (kg)
N_elevator = M_total / m_p  # number of elevator trips

# Cost parameters
c_e = 0.15  # cost per kWh (USD/kWh) - adjust as needed

# -----------------------------
# Energy and Power Calculations
# -----------------------------
def elevator_energy_per_trip():
    """Calculate total energy per elevator trip (Joules)"""
    E_gravity = (m_e + m_p) * g * h
    E_kinetic = 0.5 * (m_e + m_p) * v_mach2**2
    E_total = E_gravity + E_kinetic
    return E_total

def elevator_power():
    """Calculate required power (Watts)"""
    E_t = elevator_energy_per_trip()
    P = E_t / t_delivery
    return P

def elevator_trip_time():
    """Calculate time for one trip (hours)"""
    v_kmh = 2450  # km/h
    distance_km = 100_000  # km
    time_hours = distance_km / v_kmh
    return time_hours

def total_elevator_cost():
    """Calculate total cost of elevator transport"""
    E_t = elevator_energy_per_trip()  # Joules per trip
    E_t_kWh = E_t / (3.6e6)  # Convert Joules to kWh
    
    trip_time_hours = elevator_trip_time()  # hours per trip
    total_time_hours = N_elevator * trip_time_hours  # total operating time
    
    # Total energy consumption
    total_energy_kWh = E_t_kWh * N_elevator
    
    # Total cost
    total_cost = c_e * total_energy_kWh
    
    return total_cost

# -----------------------------
# Print Results
# -----------------------------
print("=" * 60)
print("SPACE ELEVATOR COST ANALYSIS")
print("=" * 60)
print(f"Elevator car mass: {m_e:,.0f} kg")
print(f"Payload per trip: {m_p:,.0f} kg")
print(f"Height: {h/1e6:,.0f} km")
print(f"Velocity: {v_mach2:.2f} m/s (Mach 2)")
print(f"Total mass to transport: {M_total:,.0f} kg")
print(f"Number of trips required: {N_elevator:,.0f}")
print()
print(f"Energy per trip: {elevator_energy_per_trip():.2e} J")
print(f"Energy per trip: {elevator_energy_per_trip()/3.6e6:.2e} kWh")
print(f"Power required: {elevator_power():.2e} W")
print(f"Time per trip: {elevator_trip_time():.2f} hours")
print(f"Total operating time: {N_elevator * elevator_trip_time():,.2f} hours")
print()
print(f"TOTAL ELEVATOR COST: ${total_elevator_cost():,.2f}")
print("=" * 60)

# -----------------------------
# Combined Cost Analysis
# -----------------------------
print("\n" + "=" * 60)
print("ROCKET VS ELEVATOR COMPARISON")
print("=" * 60)

elevator_cost = total_elevator_cost()

for T in [558.7, 1000, 2000, 5000]:
    rocket_cost = total_cost(T)
    total = rocket_cost + elevator_cost
    elev_pct = (elevator_cost / total) * 100
    
    print(f"\nT = {T} years:")
    print(f"  Rocket:   ${rocket_cost:.2e}")
    print(f"  Elevator: ${elevator_cost:.2e}")
    print(f"  Total:    ${total:.2e}")
    print(f"  Elevator: {elev_pct:.6f}% of total")