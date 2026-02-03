from flask import Flask, render_template, request, send_file
import pandas as pd
import io

from core.preprocessing_engine import PreprocessingEngine
from core.rule_explainer import RuleExplainer
from core.visualization_engine import VisualizationEngine

app = Flask(__name__)

# Temporary storage for cleaned data
cleaned_df_global = None


@app.route("/", methods=["GET", "POST"])
def home():
    global cleaned_df_global

    if request.method == "POST":
        uploaded_file = request.files["dataset"]
        
        filename = uploaded_file.filename.lower()

        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(uploaded_file)

            elif filename.endswith(".xlsx") or filename.endswith(".xls"):
                df = pd.read_excel(uploaded_file)

            else:
                return render_template(
                    "home.html",
                    column_names=None,
                    error="Unsupported file format. Please upload CSV or Excel file."
                )

        except pd.errors.EmptyDataError:
            return render_template(
                "home.html",
                column_names=None,
                error="Uploaded file is empty. Please upload a valid file."
            )

        except Exception as e:
            return render_template(
                "home.html",
                column_names=None,
                error=f"Error reading file: {e}"
            )

        if df.shape[1] == 0:
            return render_template(
                "home.html",
                column_names=None,
                error="Uploaded file has no columns. Please upload a valid dataset."
            )

        # ✅ NEW: extract column names
        column_names = list(df.columns)


        # -----------------------------
        # Feature Summary (Before Cleaning)
        # -----------------------------
        before_summary = {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "missing": int(df.isnull().sum().sum()),
            "duplicates": int(df.duplicated().sum())
        }

        # -----------------------------
        # Preprocessing
        # -----------------------------
        engine = PreprocessingEngine(df)

        # Target column (optional)
        target_col = request.form.get("target_column")
        if target_col and target_col in df.columns:
            engine.set_target_column(target_col)

        # -----------------------------
        # Detect user selections (STEP 1)
        # -----------------------------
        user_selected_any = any([
            request.form.get("remove_duplicates"),
            request.form.get("handle_outliers"),
            request.form.get("remove_correlation"),
            request.form.get("remove_constant_columns"),
            request.form.get("clean_text_columns")
        ])

        # Missing values are ALWAYS handled
        engine.handle_missing_values()

        # -----------------------------
        # AUTO CLEAN MODE (STEP 2)
        # -----------------------------
        if not user_selected_any:
            engine.remove_duplicates()
            engine.remove_constant_columns()
            engine.handle_outliers()
            engine.clean_text_columns()
            engine.remove_correlated_features()

            engine.actions_log.append(
                "Auto-clean mode applied: All essential cleaning steps executed."
            )

        # -----------------------------
        # USER-CONTROLLED MODE
        # -----------------------------
        else:
            if request.form.get("remove_duplicates"):
                engine.remove_duplicates()

            if request.form.get("handle_outliers"):
                engine.handle_outliers()

            if request.form.get("remove_constant_columns"):
                engine.remove_constant_columns()

            if request.form.get("clean_text_columns"):
                engine.clean_text_columns()

            if request.form.get("remove_correlation"):
                engine.remove_correlated_features()

        # -----------------------------
        # Final cleaned data
        # -----------------------------
        cleaned_df, actions = engine.get_clean_data()
        cleaned_df_global = cleaned_df

        # -----------------------------
        # Feature Summary (After Cleaning)
        # -----------------------------
        after_summary = {
            "rows": cleaned_df.shape[0],
            "columns": cleaned_df.shape[1],
            "missing": int(cleaned_df.isnull().sum().sum())
        }
        # -----------------------------
# Cleaning Impact Comparison
# -----------------------------
        rows_removed = before_summary["rows"] - after_summary["rows"]
        columns_removed = before_summary["columns"] - after_summary["columns"]

        data_reduction_percent = round(
            (rows_removed / before_summary["rows"]) * 100, 2
        ) if before_summary["rows"] > 0 else 0


        # -----------------------------
        # Explainability
        # -----------------------------
        explainer = RuleExplainer(actions)
        explanations = explainer.get_explanations()

        # -----------------------------
        # Visualization
        # -----------------------------
        viz = VisualizationEngine(df, cleaned_df)

        numeric_cols = cleaned_df.select_dtypes(include="number").columns
        selected_col = numeric_cols[0] if len(numeric_cols) > 0 else None

        hist_before_path = hist_after_path = None
        box_before_path = box_after_path = None
        heatmap_before_path = heatmap_after_path = None

        if selected_col:
            hist_before_path = viz.generate_histogram_before(selected_col)
            hist_after_path = viz.generate_histogram_after(selected_col)

            box_before_path = viz.generate_boxplot_before(selected_col)
            box_after_path = viz.generate_boxplot_after(selected_col)

            heatmap_before_path = viz.generate_correlation_heatmap_before()
            heatmap_after_path = viz.generate_correlation_heatmap_after()

        return render_template(
            "output.html",
            table=cleaned_df.to_html(classes="table", index=False),
            actions=actions,
            explanations=explanations,
            before_summary=before_summary,
            after_summary=after_summary,
            rows_removed=rows_removed,
            columns_removed=columns_removed,
            data_reduction_percent=data_reduction_percent,
            hist_before_path=hist_before_path,
            hist_after_path=hist_after_path,
            box_before_path=box_before_path,
            box_after_path=box_after_path,
            heatmap_before_path=heatmap_before_path,
            heatmap_after_path=heatmap_after_path,
            column_names=column_names 
        )

    return render_template("home.html", column_names=None)


@app.route("/download/<file_type>")
def download_file(file_type):
    global cleaned_df_global

    if cleaned_df_global is None:
        return "No data available for download", 400

    buffer = io.BytesIO()

    if file_type == "csv":
        cleaned_df_global.to_csv(buffer, index=False)
        buffer.seek(0)
        return send_file(buffer, mimetype="text/csv",
                         as_attachment=True, download_name="cleaned_data.csv")

    elif file_type == "json":
        buffer.write(cleaned_df_global.to_json(orient="records").encode())
        buffer.seek(0)
        return send_file(buffer, mimetype="application/json",
                         as_attachment=True, download_name="cleaned_data.json")

    elif file_type == "excel":
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            cleaned_df_global.to_excel(writer, index=False)
        buffer.seek(0)
        return send_file(
            buffer,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True,
            download_name="cleaned_data.xlsx"
        )

    else:
        return "Invalid file type", 400


if __name__ == "__main__":
    app.run(debug=True, port=8000)
