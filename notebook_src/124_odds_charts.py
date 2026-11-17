at_24_months = odds_results.loc["24 months"]
ranking = [at_24_months.loc["Track record", "ranking score"], at_24_months.loc["Odds model", "ranking score"], at_24_months.loc["Blend", "ranking score"], cheat_results.loc[("24 months", "Cheating model"), "ranking score"]]
ranking_names = ["Track record", "Odds model", "Blend", "Cheating model"]

groups = []
for name, (period_test, period_curves) in backtests.items():
    chance = period_curves["Blend"][:, 23]
    happened = (period_test.days_to_permit <= 720).values
    groups.append(pd.DataFrame({"odds given": chance, "happened": happened}))
pooled = pd.concat(groups)
pooled["group"] = pd.qcut(pooled["odds given"], 8)
calibration = pooled.groupby("group", observed=True).agg(given=("odds given", "mean"), happened=("happened", "mean"))

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
axes[0].bar(ranking_names, ranking, color=["gray", "#3182bd", "#2ca02c", "#d62728"])
axes[0].axhline(0.5, color="black", linestyle=":", label="Coin flip")
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("Ranking score")
axes[0].set_title("Permit within 24 months: who ranks projects best?\n(the red bar is not real skill)")
for i, value in enumerate(ranking):
    axes[0].text(i, value + 0.01, f"{value:.2f}", ha="center", fontweight="bold")
axes[0].legend()

axes[1].plot([0, 1], [0, 1], color="red", linestyle="--", label="Perfectly honest odds")
axes[1].plot(calibration.given, calibration.happened, marker="o", color="#2ca02c", linewidth=2, label="Blend, four test periods (each dot is one eighth of the projects)")
axes[1].set_xlabel("Odds the tool gave for a permit within 24 months")
axes[1].set_ylabel("Share of those projects that got one")
axes[1].set_title("Do the odds come true?")
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)
axes[1].legend()

plt.tight_layout()
plt.show()
