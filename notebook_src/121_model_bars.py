short_names = ["Typical wait\n(baseline)", "Linear\nRegression", "Neural\nNetwork", "Gradient\nBoosting", "Odds model\n(middle estimate)"]
bar_colors = ["gray", "#9ecae1", "#6baed6", "#3182bd", "#2ca02c"]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
rmse_values = day_results["RMSE (days)"].values
axes[0].bar(short_names, rmse_values, color=bar_colors)
axes[0].set_title("Root Mean Squared Error in days (lower is better)")
axes[0].set_ylabel("RMSE (days)")
for i, value in enumerate(rmse_values):
    axes[0].text(i, value + 8, f"{value:.0f}", ha="center", fontweight="bold")

r2_values = day_results["R squared"].values
axes[1].bar(short_names, r2_values, color=bar_colors)
axes[1].axhline(0, color="black", linewidth=1)
axes[1].set_title("R² score (higher is better; below 0 is worse than guessing the average)")
axes[1].set_ylabel("R² score")
for i, value in enumerate(r2_values):
    axes[1].text(i, value - 0.12 if value < 0 else value + 0.04, f"{value:.2f}", ha="center", fontweight="bold")

plt.suptitle("Days to permit, projects filed 2019 to 2021")
plt.tight_layout()
plt.show()
