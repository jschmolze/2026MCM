import math
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 1) Velocity scaling function
# ----------------------------
def velocity_from_trip_rate(trips_per_year: float, v_at_2: float, v_at_100: float) -> float:
    """
    Required max velocity as a function of trip rate (trips/year).

    Anchors:
      v(2 trips/year)   = v_at_2
      v(100 trips/year) = v_at_100

    Power-law model:
      v(rate) = v_at_2 * (rate/2)^alpha
      alpha = ln(v_at_100/v_at_2) / ln(100/2)
    """
    if trips_per_year <= 0:
        return 0.0
    if v_at_2 <= 0 or v_at_100 <= 0:
        raise ValueError("v_at_2 and v_at_100 must be > 0")

    alpha = math.log(v_at_100 / v_at_2) / math.log(100 / 2)
    return v_at_2 * (trips_per_year / 2.0) ** alpha


# ----------------------------
# 2) Energy and cost functions
# ----------------------------
def kinetic_energy_per_trip(mass_kg: float, v_m_s: float) -> float:
    """KE = 1/2 m v^2"""
    if mass_kg <= 0:
        raise ValueError("mass_kg must be > 0")
    return 0.5 * mass_kg * v_m_s**2

def total_energy_for_plan(
    total_trips: int,
    total_years: float,
    rockets: int,
    mass_kg: float,
    v_at_2: float,
    v_at_100: float
) -> dict:
    """
    Assumes:
      - trips are evenly spread across rockets and across time (constant average rate).
      - velocity depends on trip cadence PER ROCKET (since rockets are the fixed resource).

    Returns a dict with:
      trips_per_year_per_rocket, v_required, ke_per_trip, total_energy
    """
    if total_trips <= 0:
        raise ValueError("total_trips must be > 0")
    if total_years <= 0:
        raise ValueError("total_years must be > 0")
    if rockets <= 0:
        raise ValueError("rockets must be > 0")

    trips_per_year_per_rocket = total_trips / (total_years * rockets)
    v_required = velocity_from_trip_rate(trips_per_year_per_rocket, v_at_2, v_at_100)
    ke_trip = kinetic_energy_per_trip(mass_kg, v_required)
    total_energy = total_trips * ke_trip

    return {
        "trips_per_year_per_rocket": trips_per_year_per_rocket,
        "v_required": v_required,
        "ke_per_trip": ke_trip,
        "total_energy": total_energy
    }

def total_cost_for_plan(total_energy_j: float, cost_per_joule: float) -> float:
    if cost_per_joule < 0:
        raise ValueError("cost_per_joule must be >= 0")
    return total_energy_j * cost_per_joule


# -----------------------------------
# 3) Build the tradeoff curve (T vs $)
# -----------------------------------
def tradeoff_curve(
    total_trips: int = 666_667,
    rockets: int = 10,
    mass_kg: float = 1_000.0,
    v_at_2: float = 10_000.0,     # m/s at 2 trips/year (per rocket)
    v_at_100: float = 50_000.0,   # m/s at 100 trips/year (per rocket)
    cost_per_joule: float = 1e-9, # $/J
    t_min_years: float = 1.0,
    t_max_years: float = 50.0,
    points: int = 300
):
    timelines = np.linspace(t_min_years, t_max_years, points)

    costs = []
    v_requireds = []
    ke_trips = []
    rates = []

    for T in timelines:
        plan = total_energy_for_plan(
            total_trips=total_trips,
            total_years=T,
            rockets=rockets,
            mass_kg=mass_kg,
            v_at_2=v_at_2,
            v_at_100=v_at_100
        )
        cost = total_cost_for_plan(plan["total_energy"], cost_per_joule)

        costs.append(cost)
        v_requireds.append(plan["v_required"])
        ke_trips.append(plan["ke_per_trip"])
        rates.append(plan["trips_per_year_per_rocket"])

    costs = np.array(costs)
    v_requireds = np.array(v_requireds)
    ke_trips = np.array(ke_trips)
    rates = np.array(rates)

    # --- Main plot: time on x-axis, cost on y-axis (your request)
    plt.figure()
    plt.plot(timelines, costs)
    plt.xlabel("Total time to complete all trips (years)")
    plt.ylabel("Total cost (based on kinetic energy) [$]")
    plt.title("Tradeoff: Finish Faster vs Total Cost")
    plt.grid(True)

    # --- Helpful diagnostics (optional but usually worth seeing)
    plt.figure()
    plt.plot(timelines, rates)
    plt.xlabel("Total time (years)")
    plt.ylabel("Trips per year per rocket")
    plt.title("Required Cadence per Rocket vs Time")
    plt.grid(True)

    plt.figure()
    plt.plot(timelines, v_requireds)
    plt.xlabel("Total time (years)")
    plt.ylabel("Required max velocity (m/s)")
    plt.title("Required Velocity vs Time")
    plt.grid(True)

    plt.figure()
    plt.plot(timelines, ke_trips)
    plt.xlabel("Total time (years)")
    plt.ylabel("Kinetic energy per trip (J)")
    plt.title("Kinetic Energy per Trip vs Time")
    plt.grid(True)

    plt.show()

    # Print a couple example points (fast vs slow) for quick sanity checks
    for example_T in [t_min_years, (t_min_years + t_max_years)/2, t_max_years]:
        plan = total_energy_for_plan(total_trips, example_T, rockets, mass_kg, v_at_2, v_at_100)
        print(f"\n--- Example: T = {example_T:.2f} years ---")
        print(f"Trips/year/rocket: {plan['trips_per_year_per_rocket']:.2f}")
        print(f"v_required (m/s):  {plan['v_required']:.2f}")
        print(f"KE per trip (J):   {plan['ke_per_trip']:.3e}")
        print(f"Total energy (J):  {plan['total_energy']:.3e}")
        print(f"Total cost ($):    {total_cost_for_plan(plan['total_energy'], cost_per_joule):.3e}")


if __name__ == "__main__":
    # Plug your constants here.
    tradeoff_curve(
        total_trips=666_667,
        rockets=10,           # <-- FIXED number of rockets
        mass_kg=1_000.0,
        v_at_2=10_000.0,      # velocity at 2 trips/year per rocket
        v_at_100=50_000.0,    # velocity at 100 trips/year per rocket
        cost_per_joule=1e-9,
        t_min_years=2.0,
        t_max_years=50.0,
        points=300
    )
