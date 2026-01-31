import numpy as np
import matplotlib.pyplot as plt

G0 = 9.80665  # m/s^2
SECONDS_PER_YEAR = 365.25 * 24 * 3600

def delta_v_from_time_constraint(
    T_years: float,
    N_trips: int,
    distance_per_trip_m: float,
    k_impulsive: float = 2.0,
    delta_v_fixed_m_s: float = 0.0,
):
    """
    Enforces: average time budget per trip tau = T/N
    v_required = D/tau
    delta_v_time = k * v_required
    total delta_v = delta_v_fixed + delta_v_time
    """
    if T_years <= 0:
        raise ValueError("T_years must be > 0")
    if N_trips <= 0:
        raise ValueError("N_trips must be > 0")
    if distance_per_trip_m <= 0:
        raise ValueError("distance_per_trip_m must be > 0")
    if k_impulsive <= 0:
        raise ValueError("k_impulsive must be > 0")
    if delta_v_fixed_m_s < 0:
        raise ValueError("delta_v_fixed_m_s must be >= 0")

    T_seconds = T_years * SECONDS_PER_YEAR
    tau = T_seconds / N_trips  # seconds per trip
    v_required = distance_per_trip_m / tau  # m/s
    delta_v_time = k_impulsive * v_required
    delta_v_total = delta_v_fixed_m_s + delta_v_time
    return tau, v_required, delta_v_total

def propellant_per_trip(m_dry_kg: float, m_payload_kg: float, Isp_s: float, delta_v_m_s: float) -> float:
    """
    mp = mf*(exp(delta_v/(g0*Isp)) - 1)
    """
    if m_dry_kg <= 0:
        raise ValueError("m_dry_kg must be > 0")
    if m_payload_kg < 0:
        raise ValueError("m_payload_kg must be >= 0")
    if Isp_s <= 0:
        raise ValueError("Isp_s must be > 0")
    if delta_v_m_s < 0:
        raise ValueError("delta_v_m_s must be >= 0")

    mf = m_dry_kg + m_payload_kg
    return mf * (np.exp(delta_v_m_s / (G0 * Isp_s)) - 1.0)

def total_cost(N_trips: int, mp_trip_kg: float, cost_per_kg_prop: float) -> float:
    if cost_per_kg_prop < 0:
        raise ValueError("cost_per_kg_prop must be >= 0")
    return N_trips * mp_trip_kg * cost_per_kg_prop

def sweep_timeline_and_plot(
    N_trips: int = 666_667,
    # "distance per trip" is the knob that makes the time constraint meaningful
    distance_per_trip_m: float = 1e9,  # <-- set this to your mission distance per trip
    # k=2 means accelerate and then decelerate to stop
    k_impulsive: float = 2.0,
    # optional fixed mission requirement
    delta_v_fixed_m_s: float = 0.0,
    # rocket/propulsion
    m_dry_kg: float = 1000.0,
    m_payload_kg: float = 0.0,
    Isp_s: float = 350.0,
    # economics
    cost_per_kg_prop: float = 2.0,
    # timeline sweep
    T_min_years: float = 1.0,
    T_max_years: float = 50.0,
    points: int = 300,
):
    Ts = np.linspace(T_min_years, T_max_years, points)

    taus = np.zeros_like(Ts)
    v_reqs = np.zeros_like(Ts)
    dvs = np.zeros_like(Ts)
    mp_trips = np.zeros_like(Ts)
    costs = np.zeros_like(Ts)

    for i, T in enumerate(Ts):
        tau, v_req, dv = delta_v_from_time_constraint(
            T_years=T,
            N_trips=N_trips,
            distance_per_trip_m=distance_per_trip_m,
            k_impulsive=k_impulsive,
            delta_v_fixed_m_s=delta_v_fixed_m_s,
        )
        mp = propellant_per_trip(m_dry_kg, m_payload_kg, Isp_s, dv)
        cost = total_cost(N_trips, mp, cost_per_kg_prop)

        taus[i] = tau
        v_reqs[i] = v_req
        dvs[i] = dv
        mp_trips[i] = mp
        costs[i] = cost

    # Print a few reference points
    for T in [T_min_years, (T_min_years + T_max_years) / 2, T_max_years]:
        tau, v_req, dv = delta_v_from_time_constraint(
            T_years=T,
            N_trips=N_trips,
            distance_per_trip_m=distance_per_trip_m,
            k_impulsive=k_impulsive,
            delta_v_fixed_m_s=delta_v_fixed_m_s,
        )
        mp = propellant_per_trip(m_dry_kg, m_payload_kg, Isp_s, dv)
        cost = total_cost(N_trips, mp, cost_per_kg_prop)
        print(f"\n--- T = {T:.2f} years ---")
        print(f"avg time per trip tau: {tau:.6f} s")
        print(f"required speed v:      {v_req:.6f} m/s")
        print(f"delta-v per trip:      {dv:.6f} m/s")
        print(f"propellant per trip:   {mp:.6f} kg")
        print(f"total cost:            {cost:.6e} $")

    # Plots
    plt.figure()
    plt.plot(Ts, costs)
    plt.xlabel("Total timeline T (years)")
    plt.ylabel("Total propellant cost ($)")
    plt.title("Cost vs Timeline (time constraint per trip drives Δv)")
    plt.grid(True)

    plt.figure()
    plt.plot(Ts, dvs)
    plt.xlabel("Total timeline T (years)")
    plt.ylabel("Δv per trip (m/s)")
    plt.title("Δv per trip vs Timeline")
    plt.grid(True)

    plt.figure()
    plt.plot(Ts, v_reqs)
    plt.xlabel("Total timeline T (years)")
    plt.ylabel("Required speed v (m/s)")
    plt.title("Required speed vs Timeline")
    plt.grid(True)

    plt.figure()
    plt.plot(Ts, mp_trips)
    plt.xlabel("Total timeline T (years)")
    plt.ylabel("Propellant per trip (kg)")
    plt.title("Propellant per trip vs Timeline")
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    sweep_timeline_and_plot(
        N_trips=666_667,
        distance_per_trip_m=1e9,   # <-- CHANGE THIS
        k_impulsive=2.0,           # 2 = accelerate+decelerate
        delta_v_fixed_m_s=0.0,     # set >0 if you want a baseline mission Δv
        m_dry_kg=1000.0,
        m_payload_kg=0.0,
        Isp_s=350.0,
        cost_per_kg_prop=2.0,
        T_min_years=1.0,
        T_max_years=50.0,
        points=300,
    )
