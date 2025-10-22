# -*- coding: utf-8 -*-
"""
Created 2025/05/18

@author: Mario
"""
import os
import platform
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import scipy.stats as stats
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle

###############################################################################
def set_japanese_font(verbose=True):
    """
    Japanese font finder:
    Matplotlib, ReportLab compatibility
    Windows, macOS, Linux compatibility
    """

    system = platform.system()
    font_candidates = []

    if system == "Darwin":  # macOS
        font_candidates = [
            "/System/Library/Fonts/Hiragino Kaku Gothic ProN.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/YuGothic-Medium.otf",
        ]
    elif system == "Windows":
        font_candidates = [
            r"C:\Windows\Fonts\YuGothM.ttc",  # Yu Gothic Medium
            r"C:\Windows\Fonts\meiryo.ttc",
            r"C:\Windows\Fonts\msgothic.ttc",
        ]
    else:  # Linux or others
        font_candidates = [
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/takao-gothic/TakaoGothic.ttf",
        ]

    font_path = next((p for p in font_candidates if os.path.exists(p)), None)

    # === Matplotlib ===
    if font_path:
        prop = fm.FontProperties(fname=font_path)
        font_name = prop.get_name()
        plt.rcParams['font.family'] = font_name
        plt.rcParams['font.sans-serif'] = [font_name]
        plt.rcParams['axes.unicode_minus'] = False

        if verbose:
            print(f"[INFO] Matplotlib Japanese font set to: {font_name}")
            print(f"       Path: {font_path}")

        # === ReportLab ===
        try:
            pdfmetrics.registerFont(TTFont(font_name, font_path))
            if verbose:
                print(f"[INFO] ReportLab font registered: {font_name}")
        except Exception as e:
            # fallback CID font
            pdfmetrics.registerFont(UnicodeCIDFont("HeiseiMin-W3"))
            font_name = "HeiseiMin-W3"
            if verbose:
                print(f"[WARNING] Could not register system font for ReportLab ({e}), using {font_name}")

        return font_name
    else:
        # Fallback: CID universal font
        pdfmetrics.registerFont(UnicodeCIDFont("HeiseiMin-W3"))
        if verbose:
            print("[WARNING] No Japanese system font found, using HeiseiMin-W3 fallback.")
        return "HeiseiMin-W3"

###############################################################################
def save_report_to_pdf(
        filename,
        title,
        formula_image_path=None,
        comments=None,
        params=None,
        errors=None,
        figures=None,
        debug_layout=False
):
    # Font
    font_name = set_japanese_font()
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    # Layout
    margin_x = 2.0 * cm
    margin_y = 2.0 * cm
    col1_w = width - 2 * margin_x
    y_cursor = height - 2.5 * cm  # punto di partenza verticale

    # ======================================================
    # First page (ReportLab)
    # ======================================================

    # Title
    c.setFont(font_name, 22)
    c.drawCentredString(width / 2, y_cursor, title)
    y_cursor -= 1.2 * cm

    # Formula (JPG)
    if formula_image_path and os.path.exists(formula_image_path):
        try:
            img = Image.open(formula_image_path)
            img_w, img_h = img.size
            aspect = img_h / img_w
            target_w = col1_w * 0.6
            target_h = target_w * aspect
            x_pos = (width - target_w) / 2
            y_pos = y_cursor - target_h
            c.drawImage(formula_image_path, x_pos, y_pos, target_w, target_h, preserveAspectRatio=True)
            y_cursor = y_pos - 1 * cm
        except Exception as e:
            c.setFont(font_name, 10)
            c.drawString(margin_x, y_cursor, f"(Error loading formula image: {e})")
            y_cursor -= 0.8 * cm

    # Comments
    if comments:
        c.setFont(font_name, 9)
        text = c.beginText(margin_x, y_cursor)
        text.textLines(comments)
        c.drawText(text)
        y_cursor -= len(comments.split("\n")) * 0.35 * cm

    # --- Parameters table ---
    if params:
        y_cursor -= 0.5 * cm
        c.setFont(font_name, 9)
        y_cursor -= 0.6 * cm
    
        data = [[k, str(v)] for k, v in params.items()]
        table = Table(data, colWidths=[6*cm, 8*cm], rowHeights=0.4*cm)  # smaller height
        table.setStyle(TableStyle([
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
            ("FONTNAME", (0, 0), (-1, -1), font_name),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("BOX", (0, 0), (-1, -1), 0.25, colors.grey),
            # Reduce padding to shrink vertical spacing
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ]))
        table.wrapOn(c, width, height)
        table.drawOn(c, margin_x, y_cursor - len(data) * 0.4 * cm)
        y_cursor -= len(data) * 0.4 * cm + 1 * cm
    
    # --- Errors table ---
    if errors:
        y_cursor -= 0.5 * cm
        c.setFont(font_name, 9)
        y_cursor -= 0.6 * cm
    
        data = [["Error", "Description"]] + [[f"{i+1}", err] for i, err in enumerate(errors)]
        table = Table(data, colWidths=[1.5*cm, 12.5*cm], rowHeights=0.4*cm)  # compact rows
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
            ("FONTNAME", (0, 0), (-1, -1), font_name),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("BOX", (0, 0), (-1, -1), 0.25, colors.grey),
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ]))
        table.wrapOn(c, width, height)
        table.drawOn(c, margin_x, y_cursor - len(data) * 0.4 * cm)
        y_cursor -= len(data) * 0.4 * cm + 1 * cm

    # Footer
    #c.setFont(font_name, 5)
    #c.drawRightString(width - margin_x, margin_y, "rigth footer")
    #c.drawString(margin_x, margin_y, "left footer")

    # Debug: layout margins
    if debug_layout:
        c.setStrokeColorRGB(1, 0, 0)
        c.rect(margin_x, margin_y, col1_w, height - 2 * margin_y, stroke=1, fill=0)

    c.showPage()

    # ======================================================
    # Other pages (Matplotlib charts)
    # ======================================================
    if figures:
        for fig in figures:
            fig.set_size_inches(8.3, 11.7)  # formato A4
            fig.savefig("temp_fig.png", dpi=150, bbox_inches="tight")
            c.drawImage("temp_fig.png", 2 * cm, 3 * cm,
                        width - 4 * cm, height - 6 * cm,
                        preserveAspectRatio=True)
            c.showPage()
        if os.path.exists("temp_fig.png"):
            os.remove("temp_fig.png")

    c.save()
    print(f"✅ PDF salvato in: {filename}")

###############################################################################
def plot_all_feature_importances(models, target_names, top_n=20, rows_per_page=6):
    import matplotlib.pyplot as plt
    import numpy as np

    cols_per_page = 2
    plots_per_page = rows_per_page * cols_per_page
    num_targets = len(models)
    pages = []

    for start in range(0, num_targets, plots_per_page):
        end = min(start + plots_per_page, num_targets)
        sub_models = models[start:end]
        sub_targets = target_names[start:end]

        fig, axes = plt.subplots(rows_per_page, cols_per_page, figsize=(8.27, 11.69))  # A4 portrait
        axes = axes.reshape(-1, cols_per_page).flatten()

        for i, (model, target) in enumerate(zip(sub_models, sub_targets)):
            ax = axes[i]
            booster = model.booster_
            importance = booster.feature_importance(importance_type='gain')
            feature_names = booster.feature_name()

            sorted_idx = np.argsort(importance)[::-1][:top_n]
            top_features = np.array(feature_names)[sorted_idx]
            top_importance = importance[sorted_idx]

            ax.barh(
                range(top_n),
                top_importance[::-1],
                tick_label=top_features[::-1],
                color='gray'
            )
            ax.set_xlabel("Importance (gain)", fontsize=8, color='black')
            ax.set_title(f"{target}", fontsize=9, color='black')
            ax.tick_params(axis='both', which='major', labelsize=6, colors='black')

        # Rimuove gli assi non utilizzati
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        plt.close(fig)
        pages.append(fig)

    return pages

###############################################################################
def plot_error_summary_page(mae_list, rmse_list, r2_list, steps):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax1 = plt.subplots(figsize=(11.69, 8.27))  # A4 landscape

    x = np.arange(len(steps))
    width = 0.35

    # MAE and RMSE bars
    ax1.bar(x - width / 2, mae_list, width, label='MAE', color='gray')
    ax1.bar(x + width / 2, rmse_list, width, label='RMSE', color='lightgray')

    # R² line on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(x, r2_list, marker='o', color='dimgray', label='R²')
    ax2.set_ylabel('R²', fontsize=7)
    ax2.tick_params(axis='y', labelsize=6)

    ax1.set_xticks(x)
    ax1.set_xticklabels(steps, fontsize=6)
    ax1.set_title("MAE, RMSE, and R² for each Forecast Step", fontsize=8)
    ax1.set_xlabel("Forecast Step", fontsize=7)
    ax1.set_ylabel("Error", fontsize=7)
    ax1.tick_params(labelsize=6)
    ax1.grid(True)

    # Combine legends
    lines_labels = ax1.get_legend_handles_labels()
    lines_labels2 = ax2.get_legend_handles_labels()
    ax1.legend(*[sum(lol, []) for lol in zip(lines_labels, lines_labels2)], fontsize=6)

    fig.tight_layout()
    plt.close(fig)
    return fig

###############################################################################
def plot_feature_importance_heatmaps(model, feature_names, target_names, top_n=10):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from sklearn.preprocessing import MinMaxScaler

    # --------- heatmap calculation ---------
    importance_matrix = pd.DataFrame(index=feature_names)
    for i, estimator in enumerate(model.estimators_):
        importance_matrix[target_names[i]] = estimator.feature_importances_

    top_features_idx = (
        importance_matrix.mean(axis=1)
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    heatmap_data = importance_matrix.loc[top_features_idx]

    scaler = MinMaxScaler()

    # --------- Heatmap: Normalized per feature ---------
    heatmap_per_feature = pd.DataFrame(
        scaler.fit_transform(heatmap_data),
        index=heatmap_data.index,
        columns=heatmap_data.columns
    )

    # --------- Heatmap: Normalized per step ---------
    heatmap_per_step = pd.DataFrame(
        scaler.fit_transform(heatmap_data.T).T,
        index=heatmap_data.index,
        columns=heatmap_data.columns
    )

    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27))
    axes = axes.flatten()

    sns.heatmap(
        heatmap_per_feature,
        annot=True,
        annot_kws={"size": 8},
        fmt=".2f",
        cmap="Greys",
        linewidths=0.5,
        cbar_kws={'label': 'Normalized Importance (per feature)'},
        ax=axes[0]
    )
    axes[0].set_title("Normalized Feature Importance (per feature)", fontsize=8)
    axes[0].set_xlabel("Forecast Step", fontsize=7)
    axes[0].set_ylabel("Feature", fontsize=7)
    axes[0].tick_params(labelsize=6)
    axes[0].collections[0].colorbar.ax.tick_params(labelsize=6)
    axes[0].collections[0].colorbar.ax.yaxis.label.set_size(7)

    sns.heatmap(
        heatmap_per_step,
        annot=True,
        annot_kws={"size": 8},
        fmt=".2f",
        cmap="Greys",
        linewidths=0.5,
        cbar_kws={'label': 'Normalized Importance (per step)'},
        ax=axes[1]
    )
    axes[1].set_title("Normalized Feature Importance (per step)", fontsize=8)
    axes[1].set_xlabel("Target", fontsize=7)
    axes[1].set_ylabel("Feature", fontsize=7)
    axes[1].tick_params(labelsize=6)
    axes[1].collections[0].colorbar.ax.tick_params(labelsize=6)
    axes[1].collections[0].colorbar.ax.yaxis.label.set_size(7)

    fig.tight_layout()
    plt.close(fig)
    return fig

###############################################################################
def plot_feature_summary_pages(model,
                               feature_names,
                               target_names,
                               mae_list,
                               rmse_list,
                               r2_list,
                               steps,
                               has_heatmap=True,
                               top_n=10):
    pages = []

    # --------- Page 1: MAE, RMSE, R² ---------
    fig1 = plot_error_summary_page(mae_list, rmse_list, r2_list, steps)
    pages.append(fig1)

    # --------- heatmap calculation ---------
    if has_heatmap:
        fig2 = plot_feature_importance_heatmaps(model, feature_names, target_names, top_n=10)
        pages.append(fig2)

    return pages

###############################################################################
def plot_residuals_grid(y_true, y_pred, target_cols, targets_per_page=4):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    pages = []
    for start in range(0, len(target_cols), targets_per_page):
        end = start + targets_per_page
        sub_cols = target_cols[start:end]

        fig, axes = plt.subplots(len(sub_cols), 3, figsize=(11.69, 8.27))  # A4 landscape
        if len(sub_cols) == 1:
            axes = np.expand_dims(axes, axis=0)

        for i, col in enumerate(sub_cols):
            idx = start + i
            residuals = y_true[:, idx] - y_pred[:, idx]

            # Histogram in grayscale
            sns.histplot(residuals, kde=True, bins=30, ax=axes[i, 0], color='gray')
            axes[i, 0].set_title(f"{col} - Residual Histogram", fontsize=8, color='black')
            axes[i, 0].set_xlabel("Residuals", fontsize=6, color='black')
            axes[i, 0].set_ylabel("Count", fontsize=6, color='black')
            axes[i, 0].tick_params(axis='both', labelsize=5, colors='black')

            # Residuals vs Predicted in grayscale
            axes[i, 1].scatter(y_pred[:, idx], residuals, alpha=0.3, s=5, color='gray')
            axes[i, 1].axhline(0, color="black", linestyle="--", linewidth=0.8)
            axes[i, 1].set_xlabel("Predicted", fontsize=6, color='black')
            axes[i, 1].set_ylabel("Residuals", fontsize=6, color='black')
            axes[i, 1].set_title(f"{col} - Residuals vs Predicted", fontsize=8, color='black')
            axes[i, 1].tick_params(axis='both', labelsize=5, colors='black')

            # Q-Q Plot in grayscale
            stats.probplot(residuals, dist="norm", plot=axes[i, 2])
            line_fit = axes[i, 2].get_lines()[1]
            qq_points = axes[i, 2].get_lines()[0]
            qq_points.set_color("gray")
            qq_points.set_marker('o')
            qq_points.set_markersize(2)  # marker size ~ sqrt(s)
            line_fit.set_color("black")
            axes[i, 2].set_title(f"{col} - Q-Q Plot", fontsize=8, color='black')
            axes[i, 2].tick_params(axis='both', labelsize=5, colors='black')
            axes[i, 2].set_xlabel("Theoretical Quantiles", fontsize=6, color='black')
            axes[i, 2].set_ylabel("Ordered Values", fontsize=6, color='black')
            for spine in axes[i, 2].spines.values():
                spine.set_color('black')

        fig.tight_layout()
        plt.close(fig)
        pages.append(fig)

    return pages

###############################################################################
def plot_comparison_actual_predicted(y_true, y_pred, target_cols, steps, rows_per_page=3):
    import matplotlib.pyplot as plt
    import numpy as np

    plots_per_page = rows_per_page * 2
    pages = []

    for start in range(0, len(target_cols), plots_per_page):
        end = min(start + plots_per_page, len(target_cols))
        sub_cols = target_cols[start:end]
        sub_steps = steps[start:end]

        fig = plt.figure(figsize=(11.69, 8.27))
        outer_gs = GridSpec(rows_per_page, 2, figure=fig, hspace=0.5, wspace=0.3)
        
        sub_axes = []
        for row_idx in range(rows_per_page):
            row = []
            for col_idx in range(2):
                inner_gs = GridSpecFromSubplotSpec(
                    2, 1,
                    subplot_spec=outer_gs[row_idx, col_idx],
                    height_ratios=[0.8, 0.19],
                    hspace=0.01
                )
                ax_pred = fig.add_subplot(inner_gs[0])
                ax_resid = fig.add_subplot(inner_gs[1], sharex=ax_pred)
                row.append((ax_pred, ax_resid))
            sub_axes.append(row)

        for i, (col, step) in enumerate(zip(sub_cols, sub_steps)):
            row = i // 2
            col_idx = i % 2
            ax_pred, ax_resid = sub_axes[row][col_idx]

            y_true_i = y_true[:168, start + i]
            y_pred_i = y_pred[:168, start + i]
            residuals = y_true_i - y_pred_i
            std_res = np.std(residuals)

            # --- Parte alta: Previsioni ---
            ax_pred.plot(y_true_i, label='Actual', color='gray')
            ax_pred.plot(y_pred_i, label='Predicted', color='black', linestyle='--', linewidth=0.8)
            ax_pred.fill_between(np.arange(len(y_pred_i)),
                                 y_pred_i - std_res,
                                 y_pred_i + std_res,
                                 color='lightgray', alpha=0.6,
                                 label=f'±{std_res:.3f}')
            ax_pred.set_title(f'Ox(ppm) - Step {step}', fontsize=8)
            ax_pred.tick_params(axis='both', labelsize=5, colors='black')
            handles, labels = ax_pred.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax_pred.legend(by_label.values(), by_label.keys(), fontsize=5)

            # --- Parte bassa: Residui ---
            ax_resid.plot(residuals, color='gray', linewidth=0.7)
            ax_resid.axhline(std_res, color='gray', linestyle='--', linewidth=0.5)
            ax_resid.axhline(-std_res, color='gray', linestyle='--', linewidth=0.5)
            ax_resid.set_ylabel('Residuals', fontsize=6)
            ax_resid.tick_params(axis='both', labelsize=5, colors='gray')
            # Highlight ranges outside the ± std range
            out_of_band = (y_true_i < (y_pred_i - std_res)) | (y_true_i > (y_pred_i + std_res))
            in_interval = False
            start_idx = None
            for j, out in enumerate(out_of_band):
                if out and not in_interval:
                    in_interval = True
                    start_idx = j
                elif not out and in_interval:
                    in_interval = False
                    ax_resid.axvspan(start_idx, j, color='darkgrey', alpha=0.3)
            # Closes any remaining open interval
            if in_interval:
                ax_resid.axvspan(start_idx, len(out_of_band), color='darkgrey', alpha=0.3)

        # Nascondi subplot inutilizzati
        """
        total_subplots = rows_per_page * 2
        for i in range(len(sub_cols), total_subplots):
            row = i // 2
            col_idx = i % 2
            fig.delaxes(sub_axes[row, col_idx])
        """

        fig.suptitle('Comparison between actual and predicted values\nwith ± Standard Deviation Bands', fontsize=10)
        plt.close(fig)
        pages.append(fig)

    return pages

