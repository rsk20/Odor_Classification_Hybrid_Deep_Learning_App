class SimpleModelPerformanceAnalyzer:
    def __init__(self, accuracy: float, precision: float, recall: float, f1: float, model_name: str = "The model"):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.model_name = model_name

    def _performance_label(self, value: float) -> str:
        if value >= 0.9:
            return "exceptional"
        elif value >= 0.8:
            return "strong"
        elif value >= 0.7:
            return "adequate"
        elif value >= 0.6:
            return "marginal"
        return "poor"

    def generate_report(self) -> str:
        acc_label = self._performance_label(self.accuracy)
        f1_label = self._performance_label(self.f1)
        prec_label = self._performance_label(self.precision)
        recall_label = self._performance_label(self.recall)

        # Balance analysis
        diff = abs(self.precision - self.recall)
        ratio = max(self.precision, self.recall) / min(self.precision, self.recall) if min(self.precision, self.recall) > 0 else 1

        if diff < 0.05 and ratio < 1.1:
            balance = "excellent balance"
            implication = "indicating stable trade-off between precision and recall"
        elif diff < 0.1 and ratio < 1.2:
            balance = "good balance"
            implication = "with only small variation between metrics"
        else:
            balance = "noticeable imbalance"
            implication = "indicating that the model favors one metric over the other"

        # Overall conclusion
        if acc_label in ["exceptional", "strong"]:
            conclusion = (
                "This model demonstrates a highly optimized and reliable performance profile, "
                "well-suited for deployment in production environments where both accuracy and balanced class performance are essential."
            )
        elif acc_label == "adequate":
            conclusion = (
                "This model shows acceptable performance but may benefit from further tuning or data improvements "
                "before deployment."
            )
        else:
            conclusion = (
                "The model's performance is suboptimal and likely requires significant improvements "
                "before it can be considered for production use."
            )

        report = (
            f"{self.model_name} achieved {acc_label} accuracy ({self.accuracy:.2%}) and {f1_label} F1 score ({self.f1:.2%}). "
            f"Precision ({self.precision:.2%}) is {prec_label}, while recall ({self.recall:.2%}) is {recall_label}, "
            f"showing {balance} ({implication}).\n\n"
            f" {conclusion}"
        )
        return report
