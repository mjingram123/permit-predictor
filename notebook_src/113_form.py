wide = {"description_width": "230px"}
box_layout = widgets.Layout(width="460px")
class_choice = widgets.Dropdown(options=list(CATEGORY_TYPES["asset_class"].categories), value="multifamily", description="Building type", style=wide, layout=box_layout)
valuation_box = widgets.IntText(value=15000000, description="Construction cost ($)", style=wide, layout=box_layout)
stories_box = widgets.IntText(value=5, description="Stories", style=wide, layout=box_layout)
units_box = widgets.IntText(value=60, description="Housing units (0 if none)", style=wide, layout=box_layout)
area_box = widgets.IntText(value=60000, description="Floor area (square feet)", style=wide, layout=box_layout)
buildings_box = widgets.IntText(value=1, description="Buildings in the project", style=wide, layout=box_layout)
zip_box = widgets.Text(value="90015", description="ZIP code", style=wide, layout=box_layout)
zone_choice = widgets.Dropdown(options=["Typical for the ZIP code"] + list(CATEGORY_TYPES["zone_base"].categories), description="Zoning", style=wide, layout=box_layout)
approval_choice = widgets.Dropdown(options=["None (by-right project)"] + list(entitlement_times.plain_name), description="Planning approval needed", style=wide, layout=box_layout)
tick_boxes = {name: widgets.Checkbox(value=False, description=label, indent=False, layout=widgets.Layout(width="460px")) for name, label in TICK_BOXES.items()}
estimate_button = widgets.Button(description="Estimate", button_style="primary")
result_area = widgets.Output()


def on_estimate(button):
    ticked = [name for name, box in tick_boxes.items() if box.value]
    result_area.clear_output()
    with result_area:
        print_estimate(
            class_choice.value, valuation_box.value, stories_box.value, units_box.value, area_box.value,
            buildings_box.value, zip_box.value.strip()[:5], zone_choice.value, ticked, approval_choice.value,
        )


estimate_button.on_click(on_estimate)
display(widgets.VBox(
    [class_choice, valuation_box, stories_box, units_box, area_box, buildings_box, zip_box, zone_choice, approval_choice]
    + list(tick_boxes.values()) + [estimate_button, result_area]
))
