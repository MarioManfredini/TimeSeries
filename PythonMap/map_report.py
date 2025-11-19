# -*- coding: utf-8 -*-
"""
Created: 2025/06/22

Author: Mario
"""
import os
import time
import matplotlib.pyplot as plt
import numpy as np

from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


###############################################################################
def capture_html_map_screenshot(html_path, output_image="map_screenshot.jpg", delay=2, width=1024, height=768):
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument(f'--window-size={width},{height}')
    options.add_argument('--no-sandbox')
    
    driver = webdriver.Chrome(options=options)
    html_url = "file://" + os.path.abspath(html_path)
    driver.get(html_url)
    time.sleep(delay)

    screenshot_path = "_temp_screenshot.png"
    driver.save_screenshot(screenshot_path)
    driver.quit()

    # Convert to JPEG for ReportLab compatibility
    img = Image.open(screenshot_path).convert("RGB")
    img.save(output_image, "JPEG", quality=95)
    os.remove(screenshot_path)

    print(f"✅ Screenshot saved to {output_image}")
    return output_image

###############################################################################
def save_kriging_formula_as_jpg(filename="formula_kriging.jpg"):
    formula = (
        r"$\hat{z}(x_0) = \sum_{i=1}^{n} \lambda_i z(x_i)$"
    )

    explanation_lines = [
        r"$\hat{z}(x_0)$: estimated value at location $x_0$",
        r"$z(x_i)$: known value at location $x_i$",
        r"$\lambda_i$: Kriging weight for $z(x_i)$, based on spatial correlation",
        r"$\sum \lambda_i = 1$: weights sum to 1 (unbiasedness condition)",
        r"Weights depend on variogram model (e.g., exponential, spherical...)"
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')

    # Formula at the top
    ax.text(0, 1, formula, fontsize=18, ha='left', va='center')

    # Explanation lines
    y_start = 0.7
    line_spacing = 0.07
    for i, line in enumerate(explanation_lines):
        ax.text(0, y_start - i * line_spacing, line,
                fontsize=12, ha='left', va='center')

    plt.tight_layout()

    # Save temporary PNG
    temp_file = "_temp_formula_kriging.png"
    fig.savefig(temp_file, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)

    # Convert to JPEG
    img = Image.open(temp_file).convert("RGB")
    img.save(filename, format="JPEG", quality=95)
    img.close()
    os.remove(temp_file)

    print(f"✅ Saved JPEG formula image with explanation as {filename}")

###############################################################################
def plot_loocv_results(target, rmse, mae, r2, trues, preds, output_path="loocv.jpg"):
    """
    Generates and saves a combined plot:
    - Top: scatter plot (true vs predicted)
    - Bottom: line plot (true and predicted values)

    Parameters:
        trues: list or array of true values
        preds: list or array of predicted values
        output_path: path to save the resulting image
    """
    trues = np.array(trues)
    preds = np.array(preds)

    fig, axs = plt.subplots(2, 1, figsize=(6, 8), dpi=300)

    # === 1. Scatter plot ===
    axs[0].scatter(trues, preds, alpha=0.8)
    axs[0].plot([trues.min(), trues.max()], [trues.min(), trues.max()], 'r--')
    axs[0].set_xlabel(f"True {target}")
    axs[0].set_ylabel(f"Predicted {target}")
    axs[0].set_title(f"LOOCV - True vs Predicted\nRMSE={rmse:.5f}, MAE={mae:.5f}, R²={r2:.3f}")
    axs[0].grid(True)
    axs[0].axis("equal")

    # === 2. Line plot ===
    axs[1].plot(trues, label="True", color="black", linewidth=1.5)
    axs[1].plot(preds, label="Predicted", color="blue", linestyle="--")
    axs[1].set_title("True vs Predicted (Index order)")
    axs[1].set_xlabel("Index")
    axs[1].set_ylabel(target)
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"✅ LOOCV plot saved to: {output_path}")

###############################################################################
def save_model_report_pdf(
    output_path,
    table_data,
    column_headers,
    formula_image_path=None,
    map_image_path=None,
    labels_image_path=None,
    additional_image_path=None,
    title="Model Report"
):
    # === Imports ===
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import mm
    from PIL import Image
    import os

    # === Import font utilities ===
    import platform
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont

    # === Japanese-font helper ===
    def set_japanese_font(verbose=True):
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
                r"C:\Windows\Fonts\YuGothM.ttc",
                r"C:\Windows\Fonts\meiryo.ttc",
                r"C:\Windows\Fonts\msgothic.ttc",
            ]
        else:  # Linux
            font_candidates = [
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/takao-gothic/TakaoGothic.ttf",
            ]

        font_path = next((p for p in font_candidates if os.path.exists(p)), None)

        if font_path:
            prop = fm.FontProperties(fname=font_path)
            font_name = prop.get_name()

            # Matplotlib
            plt.rcParams['font.family'] = font_name
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False

            # ReportLab
            try:
                pdfmetrics.registerFont(TTFont(font_name, font_path))
            except Exception:
                pdfmetrics.registerFont(UnicodeCIDFont("HeiseiMin-W3"))
                return "HeiseiMin-W3"

            return font_name
        else:
            pdfmetrics.registerFont(UnicodeCIDFont("HeiseiMin-W3"))
            return "HeiseiMin-W3"

    # === Set Japanese PDF-friendly font ===
    jp_font = set_japanese_font(verbose=False)
    jp_font_bold = jp_font  # Some Japanese fonts have no bold variant

    # === Canvas ===
    page_width, page_height = landscape(A4)
    c = canvas.Canvas(output_path, pagesize=(page_width, page_height))
    margin = 15 * mm

    col1_w = page_width * (1 / 3)
    col2_w = page_width * (2 / 3)

    # === Title ===
    c.setFont(jp_font, 15)
    c.drawString(margin, page_height - margin, title)

    # === Formula image ===
    if formula_image_path and os.path.exists(formula_image_path):
        formula_image = Image.open(formula_image_path)
        formula_image_w, formula_image_h = formula_image.size
        formula_image_aspect = formula_image_h / formula_image_w

        image_target_width = col1_w - margin
        image_target_height = image_target_width * formula_image_aspect

        x_img = margin
        y_img = page_height - margin - image_target_height - 15 * mm

        c.drawImage(formula_image_path, x_img, y_img,
                    width=image_target_width, height=image_target_height,
                    preserveAspectRatio=True, mask='auto')
    else:
        c.setFont(jp_font, 10)
        c.drawString(margin, page_height - margin - 10 * mm,
                     "(Formula image missing)")

    # === Labels image ===
    if labels_image_path and os.path.exists(labels_image_path):
        labels_image = Image.open(labels_image_path)
        labels_image_w, labels_image_h = labels_image.size
        labels_image_aspect = labels_image_h / labels_image_w

        image_target_width = col1_w - margin
        image_target_height = image_target_width * labels_image_aspect

        x_img = margin
        y_img = margin

        c.drawImage(labels_image_path, x_img, y_img,
                    width=image_target_width, height=image_target_height,
                    preserveAspectRatio=True, mask='auto')
    else:
        c.setFont(jp_font, 10)
        c.drawString(margin, page_height / 2,
                     "(Labels image missing)")

    # === Map image (right column top) ===
    map_x = col1_w + margin
    map_max_w = col2_w - 2 * margin
    map_max_h = 80 * mm
    y_map = page_height - (margin * 1.5) - map_max_h

    if map_image_path and os.path.exists(map_image_path):
        img = Image.open(map_image_path)
        aspect = img.height / img.width
        new_w = map_max_w
        new_h = new_w * aspect
        if new_h > map_max_h:
            new_h = map_max_h
            new_w = new_h / aspect
        c.drawImage(map_image_path, map_x, y_map,
                    width=new_w, height=new_h, preserveAspectRatio=True, mask="auto")
    else:
        c.setFont(jp_font, 10)
        c.drawString(map_x, y_map + map_max_h / 2,
                     "(Map image missing)")

    # === Table (bottom right) ===
    data = [column_headers] + table_data
    col_widths = [16 * mm] * len(column_headers)

    table = Table(data, colWidths=col_widths)

    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), jp_font_bold),
        ('FONTNAME', (0, 1), (-1, -1), jp_font),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('TOPPADDING', (0, 0), (-1, -1), 1),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0.8),
        ('GRID', (0, 0), (-1, -1), 0.2, colors.black),
    ]))

    table_x = map_x
    table_y = y_map - 5 * mm

    table.wrapOn(c, 0, 0)
    table.drawOn(c, table_x, table_y - table._height)

    # === Additional image (bottom-right) ===
    if additional_image_path and os.path.exists(additional_image_path):
        img = Image.open(additional_image_path)
        img_w, img_h = img.size
        aspect = img_h / img_w

        add_img_w = map_max_w / 2
        add_img_h = add_img_w * aspect
        if add_img_h > 80 * mm:
            add_img_h = 80 * mm
            add_img_w = add_img_h / aspect

        x_add_img = map_x + map_max_w - add_img_w
        y_add_img = table_y - add_img_h
        c.drawImage(additional_image_path, x_add_img, y_add_img,
                    width=add_img_w, height=add_img_h,
                    preserveAspectRatio=True, mask='auto')
    else:
        c.setFont(jp_font, 10)
        c.drawString(map_x + map_max_w - 50 * mm,
                     margin + 5 * mm,
                     "(Additional image missing)")

    c.showPage()
    c.save()
    print(f"✅ PDF report saved to: {output_path}")

