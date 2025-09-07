import time
from typing import List, Dict, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsCollector:
    """
    Collects GA metrics across generations, provides plotting utilities,
    and final workload distribution plots for the best genome.
    """

    def __init__(self, maximize: bool = True):  # <-- default: GA maximizes
        self.generation_stats: List[Dict] = []
        self.start_time: Optional[float] = None
        self.maximize = maximize

    # -------- timing --------
    def start_timer(self):
        self.start_time = time.time()

    def stop_timer(self) -> float:
        if self.start_time is None:
            raise ValueError("Timer was not started.")
        elapsed = time.time() - self.start_time
        self.start_time = None
        return elapsed

    # -------- recording (basic/detailed) --------
    def record_generation(self, generation_num: int, fitness_values: List[float]):
        if not fitness_values:
            raise ValueError("fitness_values must be a non-empty list")
        if self.maximize:
            best = max(fitness_values)
            worst = min(fitness_values)
        else:
            best = min(fitness_values)
            worst = max(fitness_values)
        avg = sum(fitness_values) / len(fitness_values)
        self.generation_stats.append(
            {
                "generation": generation_num,
                "best_fitness": best,
                "worst_fitness": worst,
                "average_fitness": avg,
            }
        )

    def record_detailed_fitness(
        self, generation_num: int, details_list: List[Dict[str, float]]
    ):
        if not details_list:
            raise ValueError("details_list must be a non-empty list of dicts")

        # choose best/worst based on maximize flag
        def key(x):
            return x["fitness"]

        best_item = (max if self.maximize else min)(details_list, key=key)
        worst_item = (min if self.maximize else max)(details_list, key=key)
        avg_fitness = sum(d["fitness"] for d in details_list) / len(details_list)

        def avg_component(k: str) -> float:
            return sum(d.get(k, 0.0) for d in details_list) / len(details_list)

        row = {
            "generation": generation_num,
            "best_fitness": best_item["fitness"],
            "worst_fitness": worst_item["fitness"],
            "average_fitness": avg_fitness,
            "best_base_cost": best_item.get("base_cost", 0.0),
            "best_alternation_bonus": best_item.get("alternation_bonus", 0.0),
            "best_same_finger_penalty": best_item.get("same_finger_penalty", 0.0),
            "best_home_row_bonus": best_item.get("home_row_bonus", 0.0),
            "best_row_penalty": best_item.get("row_penalty", 0.0),
            "best_symbol_penalty": best_item.get("symbol_penalty", 0.0),
            "best_finger_penalty": best_item.get("finger_penalty", 0.0),
            "avg_base_cost": avg_component("base_cost"),
            "avg_alternation_bonus": avg_component("alternation_bonus"),
            "avg_same_finger_penalty": avg_component("same_finger_penalty"),
            "avg_home_row_bonus": avg_component("home_row_bonus"),
            "avg_row_penalty": avg_component("row_penalty"),
            "avg_symbol_penalty": avg_component("symbol_penalty"),
            "avg_finger_penalty": avg_component("finger_penalty"),
        }
        self.generation_stats.append(row)

    # -------- export --------
    def to_dataframe(self) -> pd.DataFrame:
        return (
            pd.DataFrame(self.generation_stats)
            .sort_values("generation")
            .reset_index(drop=True)
        )

    def save_csv(self, path: str):
        self.to_dataframe().to_csv(path, index=False)

    # -------- plotting: fitness & components --------
    def plot_fitness_curve(
        self, df=None, title="Fitness", save_path=None, as_cost: bool = False
    ):
        if df is None:
            df = self.to_dataframe()
        if df.empty:
            raise ValueError("No data to plot. Record some generations first.")

        # If you want “lower is better” visually, flip signs here
        s = -1.0 if as_cost else 1.0
        x = df["generation"]
        plt.figure(figsize=(10, 6))
        plt.plot(x, s * df["best_fitness"], label="Best Fitness")
        if "average_fitness" in df:
            plt.plot(x, s * df["average_fitness"], label="Average Fitness")
        if "worst_fitness" in df:
            plt.plot(x, s * df["worst_fitness"], label="Worst Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Cost" if as_cost else "Fitness")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()

    def plot_component_curves(
        self, df=None, use_avg=False, title="Cost Components", save_path=None
    ):
        """Plot Carpalx-style component trends for best/average individuals."""
        if df is None:
            df = self.to_dataframe()
        if df.empty:
            raise ValueError("No data to plot. Record detailed generations first.")

        prefix = "avg_" if use_avg else "best_"
        components = [
            f"{prefix}base_cost",
            f"{prefix}alternation_bonus",
            f"{prefix}same_finger_penalty",
            f"{prefix}home_row_bonus",
            f"{prefix}row_penalty",
            f"{prefix}symbol_penalty",
            f"{prefix}finger_penalty",
        ]
        components = [c for c in components if c in df.columns]

        if not components:
            raise ValueError(
                "No component columns found. Make sure to use record_detailed_fitness()."
            )

        plt.figure(figsize=(10, 6))
        for c in components:
            plt.plot(
                df["generation"],
                df[c],
                label=c.replace(prefix, "").replace("_", " ").title(),
            )
        plt.xlabel("Generation")
        plt.ylabel("Component Value")
        plt.title(title + (" (Average)" if use_avg else " (Best Per Gen)"))
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()

    # -------- plotting: workload distributions --------
    @staticmethod
    def plot_hand_fingerprint_bars(hand_counts, finger_counts, title_prefix="Layout", normalize=True):
        plot_hc = MetricsCollector._to_percent(hand_counts) if normalize else hand_counts
        plot_fc = MetricsCollector._to_percent(finger_counts) if normalize else finger_counts

        hand_keys = ["L","R"]
        hand_keys = [k for k in hand_keys if k in plot_hc] + [k for k in plot_hc if k not in hand_keys]
        finger_keys = MetricsCollector._order_fingers(list(plot_fc.keys()))

        plt.figure(figsize=(6, 4))
        vals = [plot_hc[k] for k in hand_keys]
        plt.bar(hand_keys, vals, color="skyblue")
        plt.title(f"{title_prefix} - Hand Workload")
        plt.ylabel("Percent" if normalize else "Count")
        ymax = max(vals + [0]) * 1.15
        plt.ylim(0, ymax if ymax > 0 else 1)
        for i, (k, v) in enumerate(zip(hand_keys, vals)):
            plt.text(i, max(v, 0), f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
        plt.show()

        plt.figure(figsize=(10, 4))
        fvals = [plot_fc[k] for k in finger_keys]
        plt.bar(finger_keys, fvals, color="lightgreen")
        plt.title(f"{title_prefix} - Finger Workload")
        plt.ylabel("Percent" if normalize else "Count")
        ymax = max(fvals + [0]) * 1.15
        plt.ylim(0, ymax if ymax > 0 else 1)
        for i, (k, v) in enumerate(zip(finger_keys, fvals)):
            plt.text(i, max(v, 0), f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
        plt.show()

    @staticmethod
    def plot_row_by_finger_heatmap(row_finger_counts, title="Row vs Finger Workload", normalize=True):
        # DataFrame(row->finger), transpose so index=rows, columns=fingers
        df = pd.DataFrame(row_finger_counts).T.fillna(0)

        row_order = MetricsCollector._order_rows(df.index.tolist())
        finger_order = MetricsCollector._order_fingers(df.columns.tolist())
        df = df.loc[row_order, finger_order]

        if normalize:
            total = df.values.sum() or 1.0
            df = (df / total) * 100.0
            fmt, cbar_label = ".1f", "Percent"
        else:
            fmt, cbar_label = ".0f", "Count"

        plt.figure(figsize=(10, 7))
        sns.heatmap(df, annot=True, fmt=fmt, cmap="Blues", cbar_kws={"label": cbar_label})
        plt.title(title)
        plt.ylabel("Row")
        plt.xlabel("Finger")
        plt.yticks(rotation=0)
        plt.xticks(rotation=0)
        plt.show()

 #
    # -------- helpers for workload inference --------
    @staticmethod
    def _order_fingers(keys):
        order = ["L5","L4","L3","L2","L1","R1","R2","R3","R4","R5"]
        return [k for k in order if k in keys] + [k for k in keys if k not in order]

    @staticmethod
    def _order_rows(keys):
        order = ["number", "top", "home", "bottom"]
        return [k for k in order if k in keys] + [k for k in keys if k not in order]

    @staticmethod
    def _to_percent(d):
        total = sum(d.values()) or 1.0
        return {k: (v / total) * 100.0 for k, v in d.items()}

    @staticmethod
    def _infer_hand(btn):
        if hasattr(btn, "hand") and btn.hand:
            return str(btn.hand)
        x = getattr(getattr(btn, "location", None), "x", None)
        return "L" if (x is not None and x < 5) else "R"

    @staticmethod
    def _infer_row(btn):
        if hasattr(btn, "row") and btn.row is not None:
            return str(btn.row)
        y = getattr(getattr(btn, "location", None), "y", None)
        if y is None:
            return "unknown"
        yi = int(round(y))
        mapping = {-1: "number", 0: "top", 1: "home", 2: "bottom"}
        return mapping.get(yi, str(yi))

    def _infer_finger(self, btn):
        # Prefer an explicit map if you’ve attached one: metrics.finger_map = {button_id: "L2", ...}
        if hasattr(self, "finger_map") and self.finger_map is not None:
            bid = getattr(btn, "button_id", None)
            if bid is not None and bid in self.finger_map:
                return self.finger_map[bid]
        # Otherwise use attribute or heuristic
        if hasattr(btn, "finger") and btn.finger:
            return str(btn.finger)
        x = getattr(getattr(btn, "location", None), "x", None)
        if x is None:
            return "unknown"
        if x < 5:
            if x < 1:
                return "L5"
            if x < 2:
                return "L4"
            if x < 3:
                return "L3"
            if x < 4:
                return "L2"
            return "L1"
        else:
            if x < 6:
                return "R1"
            if x < 7:
                return "R2"
            if x < 8:
                return "R3"
            if x < 9:
                return "R4"
            return "R5"

    def _compute_workloads_internal(self, layout, buttons, char_freq, bigram_freq):
        # --- compute a data-driven left/right split once ---
        xs = sorted(b.location.x for b in buttons.values())
        # choose the mid-gap between the two middle columns
        mid = len(xs) // 2
        split_x = (xs[mid-1] + xs[mid]) / 2 if len(xs) >= 2 else 5.0

        def infer_hand(btn):
            x = getattr(getattr(btn, "location", None), "x", None)
            if hasattr(btn, "hand") and btn.hand:
                return str(btn.hand)
            return "L" if (x is not None and x < split_x) else "R"

        # map characters -> buttons
        char_to_btn = {ch.character: buttons[ch.button_id] for ch in getattr(layout, "characters_set", [])}

        hand_counts, finger_counts, row_by_finger = {}, {}, {}

        for ch_obj in getattr(layout, "characters_set", []):
            ch = ch_obj.character
            freq = char_freq.get(ch, 0)
            if not freq:
                continue
            btn = char_to_btn[ch]
            hand = infer_hand(btn)
            finger = self._infer_finger(btn)   # your finger heuristic or explicit finger map
            row = self._infer_row(btn)

            hand_counts[hand] = hand_counts.get(hand, 0) + freq
            finger_counts[finger] = finger_counts.get(finger, 0) + freq
            row_by_finger.setdefault(row, {})
            row_by_finger[row][finger] = row_by_finger[row].get(finger, 0) + freq

        # same-finger bigram count
        sfb_count = 0
        for (a, b), w in bigram_freq.items():
            ba = char_to_btn.get(a)
            bb = char_to_btn.get(b)
            if ba is None or bb is None: 
                continue
            if self._infer_finger(ba) == self._infer_finger(bb):
                sfb_count += w

        return hand_counts, finger_counts, row_by_finger, sfb_count


    # -------- plotting: workload distributions (percent & ordered) --------

    def plot_final_best_layout(self, layout, buttons, char_freq, bigram_freq, run_label="Layout"):
        """Computes workload distributions and renders the three charts."""
        hand_counts, finger_counts, row_finger_counts, sfb_count = \
            self._compute_workloads_internal(layout, buttons, char_freq, bigram_freq)

        print(f"{run_label} - Same-finger bigram total: {sfb_count:.0f}")
        self.plot_hand_fingerprint_bars(hand_counts, finger_counts, title_prefix=run_label)
        self.plot_row_by_finger_heatmap(row_finger_counts, title=f"{run_label} - Row vs Finger Workload")
