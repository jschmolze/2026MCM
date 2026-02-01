"""
Water-only closed-form model for sustaining a growing lunar colony.

Assumptions implemented:
- Population ramps linearly from 0 to P_f over T years: P(t) = P_f * t / T
- Per-capita water use is constant: w_gal gallons/person/year
- Water deliveries are uniform in time over [0, T]
- Survival constraint: inventory I(t) = delivered - used >= 0 for all t in [0, T]
- Terminal reserve: at t=T, inventory >= one full year of supply at full population
- Transporting water is separate from other cargo streams
"""

from dataclasses import dataclass
import math
from typing import List, Tuple


@dataclass(frozen=True)
class WaterParams:
    P_f: float = 100_000                  # final population (persons)
    T_years: float = 10.0                 # build-out duration (years)
    w_gal_per_person_per_year: float = 100_000  # gal/person/year
    gal_to_metric_tons: float = 3.78541e-3       # metric tons/gal for water
    water_payload_per_trip_tons: float = 150.0   # metric tons/trip (water cargo)


def per_capita_use_tons_per_year(p: WaterParams) -> float:
    """Per-capita water use in metric tons/person/year."""
    return p.w_gal_per_person_per_year * p.gal_to_metric_tons


def one_year_supply_tons(p: WaterParams) -> float:
    """One-year water supply at full population (metric tons)."""
    w = per_capita_use_tons_per_year(p)
    return p.P_f * w


def population(t_years: float, p: WaterParams) -> float:
    """Population on the Moon at time t (years) under linear ramp."""
    if t_years <= 0:
        return 0.0
    if t_years >= p.T_years:
        return p.P_f
    return p.P_f * (t_years / p.T_years)


def cumulative_water_used_tons(t_years: float, p: WaterParams) -> float:
    """
    Cumulative water consumed from 0 to t (metric tons),
    W_use(t) = (w * P_f / (2T)) * t^2 for t in [0, T], then continues linearly after T.
    """
    w = per_capita_use_tons_per_year(p)
    if t_years <= 0:
        return 0.0

    if t_years <= p.T_years:
        return (w * p.P_f / (2.0 * p.T_years)) * (t_years ** 2)

    # After build-out completes: consumption continues at full population
    used_during_ramp = (w * p.P_f / (2.0 * p.T_years)) * (p.T_years ** 2)  # = w*P_f*T/2
    used_after = w * p.P_f * (t_years - p.T_years)
    return used_during_ramp + used_after


def minimum_total_water_delivered_tons(p: WaterParams) -> float:
    """
    Minimum water that must be delivered over [0, T] to:
    - cover ramp consumption AND
    - leave a one-year reserve at T.

    W_tot = w*P_f*(T/2 + 1)
    """
    w = per_capita_use_tons_per_year(p)
    return w * p.P_f * (p.T_years / 2.0 + 1.0)


def water_trips_required(p: WaterParams, ceil_to_int: bool = True) -> float:
    """Required number of water-only trips based on payload per trip."""
    W_tot = minimum_total_water_delivered_tons(p)
    trips = W_tot / p.water_payload_per_trip_tons
    return math.ceil(trips) if ceil_to_int else trips


def cumulative_water_delivered_tons(t_years: float, p: WaterParams, W_tot: float = None) -> float:
    """
    Uniform delivery over [0, T]:
    W_del(t) = (W_tot/T) * t for t in [0, T], then constant after T.
    """
    if W_tot is None:
        W_tot = minimum_total_water_delivered_tons(p)

    if t_years <= 0:
        return 0.0
    if t_years >= p.T_years:
        return W_tot
    return (W_tot / p.T_years) * t_years


def inventory_tons(t_years: float, p: WaterParams, W_tot: float = None) -> float:
    """Water inventory on the Moon at time t: I(t) = delivered - used."""
    return cumulative_water_delivered_tons(t_years, p, W_tot) - cumulative_water_used_tons(t_years, p)


def check_survival_and_reserve(p: WaterParams, grid_points: int = 200) -> Tuple[bool, float, float]:
    """
    Checks:
    - inventory >= 0 for t in [0, T] (sampled on a grid)
    - inventory(T) >= one-year supply
    Returns: (feasible, min_inventory_over_ramp, terminal_inventory_minus_reserve)
    """
    W_tot = minimum_total_water_delivered_tons(p)
    min_inv = float("inf")
    for i in range(grid_points + 1):
        t = p.T_years * (i / grid_points)
        inv = inventory_tons(t, p, W_tot)
        min_inv = min(min_inv, inv)

    terminal_inv = inventory_tons(p.T_years, p, W_tot)
    reserve = one_year_supply_tons(p)
    return (min_inv >= -1e-9 and terminal_inv + 1e-9 >= reserve, min_inv, terminal_inv - reserve)


def inventory_profile(p: WaterParams, n: int = 25) -> List[Tuple[float, float, float, float]]:
    """
    Returns a simple profile table over [0, T]:
    (t_years, population, delivered_tons, inventory_tons)
    """
    W_tot = minimum_total_water_delivered_tons(p)
    rows = []
    for i in range(n + 1):
        t = p.T_years * (i / n)
        rows.append((
            t,
            population(t, p),
            cumulative_water_delivered_tons(t, p, W_tot),
            inventory_tons(t, p, W_tot),
        ))
    return rows


if __name__ == "__main__":
    # Example usage with your numbers:
    params = WaterParams(
        P_f=100_000,
        T_years=10.0,
        w_gal_per_person_per_year=100_000,
        water_payload_per_trip_tons=150.0,
    )

    w = per_capita_use_tons_per_year(params)
    W_year = one_year_supply_tons(params)
    W_tot = minimum_total_water_delivered_tons(params)
    trips = water_trips_required(params, ceil_to_int=False)
    trips_int = water_trips_required(params, ceil_to_int=True)

    feasible, min_inv, terminal_minus_reserve = check_survival_and_reserve(params)

    print("=== Water-only model summary ===")
    print(f"Per-capita use w: {w:,.3f} metric tons/person/year")
    print(f"One-year supply at full pop: {W_year:,.0f} metric tons/year")
    print(f"Minimum total water to deliver over T: {W_tot:,.0f} metric tons")
    print(f"Water trips required (continuous): {trips:,.2f}")
    print(f"Water trips required (ceil): {trips_int:,d}")
    print(f"Survival+reserve feasible under uniform delivery: {feasible}")
    print(f"Min inventory over [0,T]: {min_inv:,.2f} metric tons")
    print(f"Terminal inventory minus reserve: {terminal_minus_reserve:,.2f} metric tons")

    # Optional: print a small profile table
    print("\nSample inventory profile (t, pop, delivered_tons, inventory_tons):")
    for t, pop, delivered, inv in inventory_profile(params, n=10):
        print(f"t={t:5.2f} yr | pop={pop:9.0f} | delivered={delivered:12.0f} | inventory={inv:12.0f}")
