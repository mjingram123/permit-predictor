fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

for city, color in [("Los Angeles", "#1f77b4"), ("Seattle", "#ff7f0e")]:
    rows = city_benchmark[city_benchmark.city == city]
    axes[0].plot(rows.filing_year, rows.share_24m * 100, marker="o", linewidth=2, color=color, label=city)
axes[0].set_title("Share of applications with a permit within 24 months")
axes[0].set_xlabel("Year the application was filed")
axes[0].set_ylabel("Share with a permit (%)")
axes[0].set_ylim(0, 100)
axes[0].legend()

months = np.arange(1, MONTHS + 1)
for name, curve in todays_track_record.items():
    axes[1].plot(months, curve * 100, linewidth=2, label=name)
axes[1].set_title("Los Angeles track record today, by building type")
axes[1].set_xlabel("Months after filing")
axes[1].set_ylabel("Share with a permit (%)")
axes[1].set_ylim(0, 100)
axes[1].legend()

plt.tight_layout()
plt.show()
