# -*- coding: utf-8 -*-
"""
Created: 2025/06/22

Author: Mario
"""

import os
import sys
import re
import pandas as pd
import folium
from datetime import datetime
from jinja2 import Template
from map_utils import generate_idw_images_by_hour

utility_dir = os.path.abspath('..\\Python\\')
if utility_dir not in sys.path:
    sys.path.append(utility_dir)
import utility


###############################################################################
def extract_labels_from_filenames(filenames):
    labels = []
    for f in filenames:
        m = re.match(r".*?(\d{4})(\d{2})(\d{2})_(\d{2})", f)
        if m:
            labels.append(f"{m.group(1)}年 {m.group(2)}月 {m.group(3)}日 {m.group(4)}時")
        else:
            labels.append("---")
    return labels

###############################################################################
def build_image_overlay_js(image_paths, bounds, opacity=0.7):
    bounds_js = f"[[{bounds[0][0]}, {bounds[0][1]}], [{bounds[1][0]}, {bounds[1][1]}]]"
    layers_js = ",".join(
        f'L.imageOverlay("{p}", {bounds_js}, {{opacity:{opacity}}})'
        for p in image_paths
    )
    return layers_js

###############################################################################
def inject_animation_and_controls(
    html_file_path,
    image_folder,
    bounds,
    ox_min,
    ox_max,
    image_prefix="ox_idw_",
    interval_ms=1000,
    opacity=0.7
):
    all_files = sorted(f for f in os.listdir(image_folder)
                       if f.endswith(".png") and f.startswith(image_prefix))
    if not all_files:
        print("⚠️ No images found.")
        return

    image_paths = [os.path.join(image_folder, f).replace("\\", "/") for f in all_files]
    labels = extract_labels_from_filenames(all_files)
    layers_js = build_image_overlay_js(image_paths, bounds, opacity)
    labels_js = "[" + ",".join(f'"{l}"' for l in labels) + "]"

    script = Template("""
<style>
.timestamp-box {
  position: absolute;
  bottom: 10px;
  right: 10px;
  background-color: rgba(255,255,255,0.8);
  padding: 8px;
  font-size: 14px;
  border-radius: 4px;
  border: 1px solid #aaa;
  z-index: 1000;
  min-width: 280px;
}
.timestamp-box span.control-btn {
  font-size: 16px;
  margin-left: 6px;
  cursor: pointer;
}
#slider {
  width: 100%;
  margin-top: 6px;
}
.scale-box {
  position: absolute;
  bottom: 10px;
  left: 10px;
  background-color: rgba(255,255,255,0.9);
  padding: 6px 10px;
  font-size: 14px;
  border-radius: 4px;
  border: 1px solid #aaa;
  z-index: 1000;
  width: 170px;
}
</style>

<div class="timestamp-box">
  <div>
    <span id="time-label">---</span>
    <span id="ctrl-play" class="control-btn">▶️</span>
    <span id="ctrl-pause" class="control-btn">⏸️</span>
  </div>
  <input type="range" id="slider" min="0" max="{{ max_index }}" value="0">
</div>

<div class="scale-box">
  <div style="font-size: 13px; font-weight: bold; text-align: center; margin-bottom: 6px;">
    Ox (ppm)
  </div>
  <div style="width: 150px; height: 12px;
              background: linear-gradient(to right, white, red);
              margin-bottom: 4px;"></div>
  <div style="display: flex; justify-content: space-between; font-size: 12px;">
    <span>{{ ox_min }}</span>
    <span>{{ ox_max }}</span>
  </div>
</div>

<script>
function getMap() {
  for (let k in window) if (k.startsWith("map_") && window[k] instanceof L.Map) return window[k];
  return null;
}
function setupAnimation() {
  const map = getMap();
  if (!map) { setTimeout(setupAnimation, 100); return; }

  const imgs = [{{ layers }}];
  const labels = {{ labels }};
  const slider = document.getElementById("slider");
  const labelDiv = document.getElementById("time-label");

  let current = 0;
  let timer = null;

  function showFrame(idx) {
    imgs.forEach((img, i) => { if (map.hasLayer(img)) map.removeLayer(img); });
    map.addLayer(imgs[idx]);
    labelDiv.innerText = labels[idx];
    slider.value = idx;
  }

  function startInterval() {
    if (timer) return;
    timer = setInterval(() => {
      current = (current + 1) % imgs.length;
      showFrame(current);
    }, {{ interval }});
  }

  function stopInterval() {
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
  }

  document.getElementById("ctrl-play").onclick = () => startInterval();
  document.getElementById("ctrl-pause").onclick = () => stopInterval();

  slider.oninput = function () {
    stopInterval();
    current = parseInt(this.value);
    showFrame(current);
  };

  showFrame(0);
  startInterval();
}
setupAnimation();
</script>
""")

    html_script = script.render(
        layers=layers_js,
        labels=labels_js,
        interval=interval_ms,
        max_index=len(labels) - 1,
        ox_min=f"{ox_min:.3f}",
        ox_max=f"{ox_max:.3f}"
    )

    with open(html_file_path, "r", encoding="utf-8") as f:
        html = f.read()

    if "</body>" in html:
        html = html.replace("</body>", html_script + "\n</body>")
    else:
        html += html_script

    with open(html_file_path, "w", encoding="utf-8") as f:
        f.write(html)

    print("✅ Animation and controls successfully injected.")


# === CONFIG ===
data_dir = '..\\data\\Osaka\\'
prefecture_code = '27'
station_coordinates = 'Stations_Ox.csv'
csv_path = os.path.join(data_dir, station_coordinates)

stations_df = pd.read_csv(csv_path, skipinitialspace=True)
from_datetime = datetime(2025, 5, 1, 0)

records = utility.load_ox_time_series(
    data_dir=data_dir,
    stations_df=stations_df,
    from_datetime=from_datetime,
    prefecture_code=prefecture_code,
    hours=48
)

# === Folium Map ===
center_lat = sum(r['latitude'] for r in records) / len(records)
center_lon = sum(r['longitude'] for r in records) / len(records)
m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

# Save map
html_file_name = "OxMapIDW_LastTwoDays.html"
m.save(html_file_name)
print(f"Map saved to: {html_file_name}")

bounds, ox_min, ox_max = generate_idw_images_by_hour(
    records,
    k=7,
    power=1.0,
    output_dir="idw_frames",
    num_cells=800,
    overwrite=True
)

inject_animation_and_controls(
    html_file_path=html_file_name,
    image_folder="idw_frames",
    bounds=bounds,
    ox_min=ox_min,
    ox_max=ox_max,
    interval_ms=1000
)
