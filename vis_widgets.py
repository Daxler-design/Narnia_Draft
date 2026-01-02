import open3d.visualization.gui as gui

def create_slider_row(label_text, min_val, max_val, init_val, on_change_callback, is_int=False):
    v = gui.Vert(0, gui.Margins(0, 0, 0, 0))
    v.add_child(gui.Label(label_text))
    h = gui.Horiz(5)
    
    num_edit = gui.NumberEdit(gui.NumberEdit.INT if is_int else gui.NumberEdit.DOUBLE)
    slider = gui.Slider(gui.Slider.INT if is_int else gui.Slider.DOUBLE)
    slider.set_limits(min_val, max_val)

    if is_int:
        num_edit.int_value = int(init_val)
        slider.int_value = int(init_val)
    else:
        num_edit.double_value = float(init_val)
        slider.double_value = float(init_val)

    def on_slider(val):
        if is_int:
            num_edit.int_value = int(val)
        else:
            num_edit.double_value = float(val)
        if on_change_callback:
            on_change_callback(val)

    def on_edit(val):
        if is_int:
            slider.int_value = int(val)
        else:
            slider.double_value = float(val)
        if on_change_callback:
            on_change_callback(val)

    slider.set_on_value_changed(on_slider)
    num_edit.set_on_value_changed(on_edit)
    
    h.add_child(num_edit)
    h.add_child(slider)
    v.add_child(h)
    return v, slider, num_edit

def create_file_input_row(label_text, default_path, on_browse_callback):
    v = gui.Vert(0, gui.Margins(0, 0, 0, 0))
    v.add_child(gui.Label(label_text))
    h = gui.Horiz(4)
    
    tedit = gui.TextEdit()
    tedit.text_value = default_path
    
    btn = gui.Button("...")
    btn.horizontal_padding_em = 0.5
    btn.set_on_clicked(on_browse_callback)
    
    h.add_child(tedit)
    h.add_child(btn)
    v.add_child(h)
    return v, tedit
