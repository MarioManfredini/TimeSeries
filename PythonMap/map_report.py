# -*- coding: utf-8 -*-
"""
Created: 2025/06/22

Author: Mario
"""
import os
import time
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from PIL import Image
import matplotlib.pyplot as plt
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

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

def save_idw_formula_as_jpg(filename="formula_idw.jpg"):
    formula = (
        r"$\hat{z}(x_0) = \frac{\sum_{i=1}^{k} w_i z_i}"
        r"{\sum_{i=1}^{k} w_i}, \quad \text{where } w_i = \frac{1}{d(x_0, x_i)^p}$"
    )

    explanation_lines = [
        r"$x_0$: location to interpolate",
        r"$x_i$: known data point location",
        r"$z_i$: known value at $x_i$",
        r"$d(x_0, x_i)$: distance between $x_0$ and $x_i$",
        r"$w_i$: weight of $z_i$",
        r"$p$: power parameter (controls weight decay)",
        r"$k$: number of nearest neighbors"
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')

    # Formula centered on top
    ax.text(0, 1, formula, fontsize=18, ha='left', va='center')

    # Explanation aligned to left below
    y_start = 0.7
    line_spacing = 0.07
    for i, line in enumerate(explanation_lines):
        ax.text(0, y_start - i * line_spacing, line,
                fontsize=12, ha='left', va='center')

    plt.tight_layout()

    temp_file = "_temp_formula.png"
    fig.savefig(temp_file, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)

    img = Image.open(temp_file).convert("RGB")
    img.save(filename, format="JPEG", quality=95)
    img.close()

    os.remove(temp_file)
    print(f"✅ Saved JPEG formula image with explanation as {filename}")


def save_idw_report_pdf(
    output_path,
    results,
    formula_image_path='formula_idw.jpg',
    html_screenshot_path='map_screenshot.jpg',
    idw_labels_image_path='ox_idw_labels.png',
    title="IDW Cross-validation Report"
):
    page_width, page_height = landscape(A4)
    c = canvas.Canvas(output_path, pagesize=(page_width, page_height))
    margin = 15 * mm
    col_split = page_width / 2  # 50% table, 50% image

    # 1/3 for text/metrics, 2/3 for map
    col1_w = page_width * (1 / 3)
    col2_w = page_width * (2 / 3)

    # === Title ===
    c.setFont("Helvetica-Bold", 15)
    c.drawString(margin, page_height - margin, title)

    # === Formula image ===
    if formula_image_path and os.path.exists(formula_image_path):
        formula_w = col_split - 2 * margin
        formula_h = 40 * mm
        c.drawImage(formula_image_path, margin, page_height - margin - 45 * mm,
                    width=formula_w, height=formula_h, preserveAspectRatio=True, mask='auto')
    else:
        c.setFont("Helvetica", 10)
        c.drawString(margin, page_height - margin - 10 * mm, "(Formula image missing)")

    # === IDW image with station labels ===
    if idw_labels_image_path and os.path.exists(idw_labels_image_path):
        img = Image.open(idw_labels_image_path)
        img_w, img_h = img.size
        aspect = img_h / img_w

        image_target_width = col1_w - 2 * margin
        image_target_height = image_target_width * aspect

        x_img = margin
        y_img = (page_height - image_target_height) / 2

        c.drawImage(idw_labels_image_path, x_img, y_img,
                    width=image_target_width, height=image_target_height,
                    preserveAspectRatio=True, mask='auto')
    else:
        c.setFont("Helvetica", 10)
        c.drawString(col_split + margin, page_height / 2, "(Map image missing)")

    # === Map image (screenshot) on right 2/3 ===
    map_x = col1_w + margin
    map_max_w = col2_w - 2 * margin
    map_max_h = 90 * mm
    y_map = page_height - margin - map_max_h
    if html_screenshot_path and os.path.exists(html_screenshot_path):
        img = Image.open(html_screenshot_path)
        aspect = img.height / img.width
        new_w = map_max_w
        new_h = new_w * aspect
        if new_h > map_max_h:
            new_h = map_max_h
            new_w = new_h / aspect
        c.drawImage(html_screenshot_path, map_x, y_map,
                    width=new_w, height=new_h, preserveAspectRatio=True, mask="auto")
    else:
        c.setFont("Helvetica", 10)
        c.drawString(col_split + margin, page_height / 2, "(Screenshot missing)")

    # === Table ===
    table_x = map_x
    table_y = y_map - 5 * mm
    c.setFont("Helvetica-Bold", 8)
    c.drawString(table_x, table_y, "k   p    RMSE       MAE       R²")
    c.setFont("Helvetica", 7)
    for i, (k, power, rmse, mae, r2) in enumerate(results):
        y = table_y - (i + 1) * 4 * mm
        line = f"{k:<2d}  {power:<5.2f}   {rmse:<8.5f}  {mae:<8.5f}  {r2:<6.3f}"
        c.drawString(table_x, y, line)

    c.showPage()
    c.save()
    print(f"✅ PDF report saved to: {output_path}")

