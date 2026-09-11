import os
import pya


gds_path = os.environ["BCUDA_GDS"]
png_path = os.environ["BCUDA_LAYOUT_PNG"]

view = pya.LayoutView()
view.load_layout(gds_path, 0)
view.max_hier()
view.zoom_fit()
view.save_image(png_path, 1800, 1800)
