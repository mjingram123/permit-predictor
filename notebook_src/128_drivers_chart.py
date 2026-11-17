recent_projects = projects[projects.filed >= DATA_DATE - pd.Timedelta(days=5 * 365)]
effects = {}
for name, label in TICK_BOXES.items():
    switched_on = recent_projects.copy()
    switched_on[name] = 1
    switched_off = recent_projects.copy()
    switched_off[name] = 0
    effects[label] = (permit_curves(final_model, switched_on)[:, 23] - permit_curves(final_model, switched_off)[:, 23]).mean() * 100
effects = pd.Series(effects).sort_values()

plt.figure(figsize=(10, 6))
plt.barh(effects.index, effects.values, color=np.where(effects.values < 0, "#d62728", "#2ca02c"))
plt.axvline(0, color="black", linewidth=1)
plt.xlabel("Change in 24-month permit odds (percentage points)")
plt.title("How each project feature moves the odds")
plt.tight_layout()
plt.show()
