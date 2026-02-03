import matplotlib.pyplot as plt
import seaborn as sns
import os


class VisualizationEngine:
    """
    Generates visualizations to compare data
    before and after preprocessing.
    """

    def __init__(self, before_df, after_df):
        self.before = before_df
        self.after = after_df

        # Ensure static folder exists
        self.output_dir = "static/plots"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate_histogram_before(self, column):
        plt.figure()
        sns.histplot(self.before[column], kde=True, color="red")
        plt.title(f"Histogram of {column} (Before Cleaning)")
        plt.xlabel(column)
        plt.ylabel("Count")
        path = os.path.join(self.output_dir, "histogram_before.png")
        plt.savefig(path)
        plt.close()

        return "static/plots/histogram_before.png"
    def generate_histogram_after(self, column):
        plt.figure()
        sns.histplot(self.after[column], kde=True, color="green")
        plt.title(f"Histogram of {column} (After Cleaning)")
        plt.xlabel(column)
        plt.ylabel("Count")

        path = os.path.join(self.output_dir, "histogram_after.png")
        plt.savefig(path)
        plt.close()

        return "static/plots/histogram_after.png"
    def generate_boxplot_before(self, column):
        plt.figure()
        sns.boxplot(x=self.before[column], color="red")
        plt.title(f"Box Plot of {column} (Before Cleaning)")

        path = os.path.join(self.output_dir, "boxplot_before.png")
        plt.savefig(path)
        plt.close()

        return "static/plots/boxplot_before.png"


    def generate_boxplot_after(self, column):
        plt.figure()
        sns.boxplot(x=self.after[column], color="green")
        plt.title(f"Box Plot of {column} (After Cleaning)")

        path = os.path.join(self.output_dir, "boxplot_after.png")
        plt.savefig(path)
        plt.close()

        return "static/plots/boxplot_after.png"


    def generate_correlation_heatmap_before(self):
        plt.figure(figsize=(8, 6))
        corr_before = self.before.select_dtypes(include="number").corr()
        sns.heatmap(corr_before, cmap="coolwarm")
        plt.title("Correlation Heatmap (Before Cleaning)")

        path = os.path.join(self.output_dir, "heatmap_before.png")
        plt.savefig(path)
        plt.close()

        return "static/plots/heatmap_before.png"


    def generate_correlation_heatmap_after(self):
        plt.figure(figsize=(8, 6))
        corr_after = self.after.select_dtypes(include="number").corr()
        sns.heatmap(corr_after, cmap="coolwarm")
        plt.title("Correlation Heatmap (After Cleaning)")

        path = os.path.join(self.output_dir, "heatmap_after.png")
        plt.savefig(path)
        plt.close()

        return "static/plots/heatmap_after.png"

