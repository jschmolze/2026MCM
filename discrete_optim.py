import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = "transportation_costs_comparison.csv"

def optimize_T1_T2(
    df: pd.DataFrame,
    T: int,
    two_modes: bool = True,
):
    """
    Minimize C_A(T1) + C_B(T2); where C_A(T1) and C_B(T2) is the total cost for each mode 
    and subject to T1 + T2 = T
    using discrete (integer-year) values from the CSV.

    @param df dataframe -> consisting of historical (simulation) costs of the two modes
    @param T int -> desired timeline duration
    @param two_modes boolean -> True if both modes are being considered, false otherwise

    @ returns -> dict with best split and costs, plus a DataFrame of all feasible candidates.
    """
    if not isinstance(T, (int, np.integer)):
        raise TypeError("T must be an integer number of years.")

    colA, colB = "Model_1a_Total", "Model_1b_Total"
    A = df.set_index("T_years")[colA].to_dict()
    B = df.set_index("T_years")[colB].to_dict()

    # Elevator cost (Assumed constant every year)
    E = float(df["Elevator_Cost"].dropna().iloc[0]) if "Elevator_Cost" in df.columns else 0.0

    possible_splits = []
    # Enumerate all possible integer splits combination
    for T1 in range(0, T + 1):
        T2 = T - T1

        cA = A.get(T1, np.nan)
        cB = B.get(T2, np.nan)

        if two_modes:
            if np.isnan(cA) or np.isnan(cB):
                continue
        else:
            if np.isnan(cA) and np.isnan(cB):
                continue
            cA = 0.0 if np.isnan(cA) else float(cA)
            cB = 0.0 if np.isnan(cB) else float(cB)

        obj = cA + cB
        possible_splits.append((T1, T2, cA, cB, obj))

    if not possible_splits:
        raise ValueError(
            f"No feasible split found for T={T}. "
            f"(Check that T is within the table range and that both modes exist for T1 and T2.)"
        )

    cand_df = pd.DataFrame(
        possible_splits,
        columns=["T1", "T2", f"{colA}", f"{colB}", "objective_sum"]
    ).sort_values("objective_sum", ascending=True, ignore_index=True)

    best = cand_df.iloc[0].to_dict()

    # If minimizing rocket costs and elevator should be counted once, report total = objective + E
    best["elevator_cost_counted_once"] = 0.0
    best["total_cost_with_elevator_once"] = best["objective_sum"]

    # Add percent-of-time split
    best["pct_time_mode_A"] = 100.0 * best["T1"] / T
    best["pct_time_mode_B"] = 100.0 * best["T2"] / T

    return best, cand_df


if __name__ == "__main__":
    df = pd.read_csv(path)

    if 0 not in df["T_years"].values:
        row0 = df.iloc[0].copy()
        row0["T_years"] = 0

        for c in ["Model_1a_Total", "Model_1b_Total",
                  "Model_1a_Rocket_Cost", "Model_1b_Rocket_Cost"]:
            if c in df.columns:
                row0[c] = 0.0

        # Elevator cost is constant
        df = pd.concat([pd.DataFrame([row0]), df], ignore_index=True)
        df = df.sort_values("T_years").reset_index(drop=True)

    # Randomized Deadline
    T = [200, 300,500, 743, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2200, 2400, 2600, 2700, 2900, 3100]
    
    result_a = []
    result_b = []

    for t in T:
        print(t)
        best, all_candidates = optimize_T1_T2(
            df, t,
            two_modes=True
        )

        #Result array for % based on different timelines
        result_a.append(best['pct_time_mode_A'])
        result_b.append(best['pct_time_mode_B'])   

        print(f"Result for T={t}:")
        print(f"  T1 (Mode A) = {best['T1']} years ({best['pct_time_mode_A']:.2f}%)")
        print(f"  T2 (Mode B) = {best['T2']} years ({best['pct_time_mode_B']:.2f}%)")
        print(f"  Total Cost (A+B) = {best['objective_sum']:.6e}")
        print(f"  Total with elevator = {best['total_cost_with_elevator_once']:.6e}")

        # Optional: see the top 10 candidate splits
        print("\nTop 10 candidates:")
        print(all_candidates.head(10).to_string(index=False))


    plt.plot(T, result_a, color='red', label='Mode A')
    plt.plot(T, result_b, color='green', label='Mode B')
    plt.title('Usage % using Mode vs T')
    plt.ylabel('% of T')
    plt.xlabel('Timeline [T]')
    plt.grid(True)
    plt.legend(loc='upper right')

    plt.show()

    

