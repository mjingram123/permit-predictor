fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.scatterplot(x=y_test, y=odds_days, alpha=0.4, ax=axes[0], color="teal")
ideal_line = np.linspace(0, MONTHS * 30, 100)
axes[0].plot(ideal_line, ideal_line, color="red", linestyle="--", label="Perfect prediction")
axes[0].set_title("Actual vs. predicted days to permit (odds model, middle estimate)")
axes[0].set_xlabel("Actual days to permit")
axes[0].set_ylabel("Predicted days to permit")
axes[0].legend()

residuals = y_test - odds_days
sns.histplot(residuals, kde=True, ax=axes[1], color="purple")
axes[1].axvline(x=0, color="red", linestyle="--")
axes[1].set_title("Distribution of prediction errors")
axes[1].set_xlabel("Actual minus predicted (days)")
axes[1].set_ylabel("Projects")

plt.tight_layout()
plt.show()
