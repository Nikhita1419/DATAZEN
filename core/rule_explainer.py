class RuleExplainer:
    """
    This class is responsible for generating
    human-readable explanations for each
    preprocessing action applied to the dataset.
    """

    def __init__(self, action_logs):
        self.action_logs = action_logs

    def get_explanations(self):
        explanation_list = []

        for action in self.action_logs:

            if "Missing values" in action:
                explanation_list.append(
                    "The dataset contained missing values. "
                    "These were replaced using statistical measures "
                    "to maintain completeness and avoid data loss."
                )

            elif "Duplicate rows" in action:
                explanation_list.append(
                    "Duplicate records were detected and removed "
                    "to prevent repeated data from biasing analysis results."
                )

            elif "Outliers" in action:
                explanation_list.append(
                    "Outliers were identified using the IQR method "
                    "and removed to reduce noise and improve data consistency."
                )

            elif "correlated" in action.lower():
                explanation_list.append(
                    "Highly correlated columns were removed "
                    "to reduce redundancy and improve feature independence."
                )

            else:
                explanation_list.append(
                    "A data preprocessing operation was applied "
                    "to improve overall data quality."
                )

        return explanation_list
