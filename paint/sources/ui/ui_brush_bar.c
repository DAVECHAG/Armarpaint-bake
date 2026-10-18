
#include "../global.h"

// Brush toolbar on the left of the tool header: advanced options, primary and secondary
// colors, brush mask and a live stroke preview. The primary color is the Color node plugged
// into Base Color of the active material, the material stays the source of what is painted.

bool ui_brush_bar_advanced = false; // Drawing the advanced options popup

static char *ui_brush_bar_swap_key();

static u32              ui_brush_bar_secondary = 0xffffffff;
static bool             ui_brush_bar_edit_secondary;
static u32              ui_brush_bar_edit_color;
static ui_color_state_t ui_brush_bar_color_state;

static gpu_texture_t *ui_brush_bar_colors_image  = NULL;
static u32            ui_brush_bar_colors_key[2] = {0, 0};
static gpu_texture_t *ui_brush_bar_preview_image = NULL;
static f32            ui_brush_bar_preview_key[6];
static gpu_texture_t *ui_brush_bar_round_image = NULL;

#define UI_BRUSH_BAR_COLORS_W  40
#define UI_BRUSH_BAR_COLORS_H  32
#define UI_BRUSH_BAR_PREVIEW_W 128
#define UI_BRUSH_BAR_PREVIEW_H 32

// Color node driving Base Color of the active material, NULL when the base color comes from elsewhere
static ui_node_t *ui_brush_bar_color_node() {
	if (g_context->material == NULL) {
		return NULL;
	}
	ui_node_canvas_t *canvas = g_context->material->canvas;
	ui_node_t        *nout   = NULL;
	for (i32 i = 0; i < canvas->nodes->length; ++i) {
		if (string_equals(canvas->nodes->buffer[i]->type, "OUTPUT_MATERIAL_PBR")) {
			nout = canvas->nodes->buffer[i];
			break;
		}
	}
	if (nout == NULL) {
		return NULL;
	}
	for (i32 i = 0; i < canvas->links->length; ++i) {
		ui_node_link_t *l = canvas->links->buffer[i];
		if (l->to_id == nout->id && l->to_socket == 0) {
			ui_node_t *n = ui_get_node(canvas->nodes, l->from_id);
			return n != NULL && string_equals(n->type, "RGB") ? n : NULL;
		}
	}
	return NULL;
}

static u32 ui_brush_bar_get_primary() {
	ui_node_t *n = ui_brush_bar_color_node();
	if (n == NULL) {
		return 0xff808080;
	}
	f32 *v = n->outputs->buffer[0]->default_value->buffer;
	return color_from_floats(v[0], v[1], v[2], 1.0);
}

static void ui_brush_bar_set_primary(u32 c) {
	ui_node_t *n = ui_brush_bar_color_node();
	if (n == NULL) {
		return;
	}
	f32 *v = n->outputs->buffer[0]->default_value->buffer;
	v[0]   = color_get_rb(c) / 255.0;
	v[1]   = color_get_gb(c) / 255.0;
	v[2]   = color_get_bb(c) / 255.0;
	make_material_parse_paint_material(true);
	util_render_make_material_preview();
	ui_dock_redraw(1);
	if (ui_nodes_hwnd != NULL) {
		ui_nodes_hwnd->redraws = 2;
	}
}

static void ui_brush_bar_swap() {
	if (ui_brush_bar_color_node() == NULL) {
		return;
	}
	u32 primary            = ui_brush_bar_get_primary();
	ui_brush_bar_set_primary(ui_brush_bar_secondary);
	ui_brush_bar_secondary = primary;
	ui_header_handle->redraws = 2;
}

static gpu_texture_t *ui_brush_bar_texture(u8 *pixels, i32 w, i32 h, gpu_texture_t *old) {
	if (old != NULL) {
		gpu_delete_texture(old);
	}
	buffer_t b = {.buffer = pixels, .length = w * h * 4};
	return gpu_create_texture_from_bytes(&b, w, h, GPU_TEXTURE_FORMAT_RGBA32);
}

static void ui_brush_bar_put(u8 *p, i32 w, i32 x, i32 y, u32 c) {
	u8 *o = p + (y * w + x) * 4;
	o[0]  = color_get_rb(c);
	o[1]  = color_get_gb(c);
	o[2]  = color_get_bb(c);
	o[3]  = color_get_ab(c);
}

static void ui_brush_bar_square(u8 *p, i32 w, i32 x0, i32 y0, i32 s, u32 c) {
	for (i32 y = y0; y < y0 + s; ++y) {
		for (i32 x = x0; x < x0 + s; ++x) {
			bool border = x == x0 || y == y0 || x == x0 + s - 1 || y == y0 + s - 1;
			ui_brush_bar_put(p, w, x, y, border ? 0xffd0d0d0 : c);
		}
	}
}

// Two overlapping squares, primary on top, and a swap arrow in the top right corner
static void ui_brush_bar_update_colors(u32 primary, u32 secondary) {
	if (ui_brush_bar_colors_image != NULL && ui_brush_bar_colors_key[0] == primary && ui_brush_bar_colors_key[1] == secondary) {
		return;
	}
	ui_brush_bar_colors_key[0] = primary;
	ui_brush_bar_colors_key[1] = secondary;
	i32 w = UI_BRUSH_BAR_COLORS_W, h = UI_BRUSH_BAR_COLORS_H;
	u8 *p = calloc(w * h * 4, 1);
	ui_brush_bar_square(p, w, 15, 11, 20, secondary);
	ui_brush_bar_square(p, w, 2, 2, 20, primary);
	// Swap arrow: quarter circle from top to right with two arrow heads
	u32 a = 0xffe0e0e0;
	for (i32 i = 0; i <= 8; ++i) {
		f32 t = i / 8.0 * 1.5707963;
		ui_brush_bar_put(p, w, 26 + (i32)(cosf(t) * 8.0 + 0.5) - 1, 10 - (i32)(sinf(t) * 8.0 + 0.5), a);
	}
	ui_brush_bar_put(p, w, 26, 0, a);
	ui_brush_bar_put(p, w, 26, 3, a);
	ui_brush_bar_put(p, w, 32, 9, a);
	ui_brush_bar_put(p, w, 35, 9, a);
	ui_brush_bar_colors_image = ui_brush_bar_texture(p, w, h, ui_brush_bar_colors_image);
	free(p);
}

// Stroke drawn with the current radius, hardness, opacity and color
static void ui_brush_bar_update_preview(u32 color) {
	bool eraser = g_context->tool == TOOL_TYPE_ERASER;
	f32  key[6] = {g_context->brush_radius, g_context->brush_hardness, g_context->brush_opacity, (f32)color, eraser ? 1.0 : 0.0,
	               (f32)(uintptr_t)g_context->brush_mask_image};
	if (ui_brush_bar_preview_image != NULL && memcmp(key, ui_brush_bar_preview_key, sizeof(key)) == 0) {
		return;
	}
	memcpy(ui_brush_bar_preview_key, key, sizeof(key));

	i32 w = UI_BRUSH_BAR_PREVIEW_W, h = UI_BRUSH_BAR_PREVIEW_H;
	u8 *p = malloc(w * h * 4);
	f32 r = math_min(math_max(2.0 + g_context->brush_radius * 9.0, 2.0), h * 0.45);
	f32 hard = math_min(g_context->brush_hardness, 0.999);
	u32 c    = eraser ? 0xffe8e8e8 : color;
	f32 cr = color_get_rb(c), cg = color_get_gb(c), cb = color_get_bb(c);

	// Wave through the middle of the image
	f32 pts[64][2];
	for (i32 i = 0; i < 64; ++i) {
		f32 t     = i / 63.0;
		pts[i][0] = r + 2.0 + t * (w - 2.0 * r - 4.0);
		pts[i][1] = h * 0.5 + sinf(t * 6.2831853) * (h * 0.5 - r - 2.0);
	}
	for (i32 y = 0; y < h; ++y) {
		for (i32 x = 0; x < w; ++x) {
			f32 d = 1e9;
			for (i32 i = 0; i < 63; ++i) {
				f32 ax = pts[i][0], ay = pts[i][1];
				f32 bx = pts[i + 1][0] - ax, by = pts[i + 1][1] - ay;
				f32 t  = ((x + 0.5 - ax) * bx + (y + 0.5 - ay) * by) / (bx * bx + by * by);
				t      = math_min(math_max(t, 0.0), 1.0);
				f32 dx = x + 0.5 - (ax + bx * t), dy = y + 0.5 - (ay + by * t);
				d      = math_min(d, sqrtf(dx * dx + dy * dy));
			}
			f32 u = d / r;
			f32 a = u >= 1.0 ? 0.0 : u <= hard ? 1.0 : 1.0 - (u - hard) / (1.0 - hard);
			a *= g_context->brush_opacity;
			f32 bg = 34.0;
			u8 *o  = p + (y * w + x) * 4;
			o[0]   = (u8)(bg + (cr - bg) * a);
			o[1]   = (u8)(bg + (cg - bg) * a);
			o[2]   = (u8)(bg + (cb - bg) * a);
			o[3]   = 255;
		}
	}
	ui_brush_bar_preview_image = ui_brush_bar_texture(p, w, h, ui_brush_bar_preview_image);
	free(p);
}

// Soft white disc shown when the brush has no mask
static gpu_texture_t *ui_brush_bar_round() {
	if (ui_brush_bar_round_image != NULL) {
		return ui_brush_bar_round_image;
	}
	i32 s = 32;
	u8 *p = malloc(s * s * 4);
	for (i32 y = 0; y < s; ++y) {
		for (i32 x = 0; x < s; ++x) {
			f32 dx = x + 0.5 - s * 0.5, dy = y + 0.5 - s * 0.5;
			f32 d  = sqrtf(dx * dx + dy * dy) / (s * 0.38);
			f32 a  = math_min(math_max((1.0 - d) * 6.0, 0.0), 1.0);
			u8  v  = (u8)(a * 255.0);
			u8 *o  = p + (y * s + x) * 4;
			o[0] = o[1] = o[2] = v;
			o[3]               = 255;
		}
	}
	ui_brush_bar_round_image = ui_brush_bar_texture(p, s, s, NULL);
	free(p);
	return ui_brush_bar_round_image;
}

// Brush mask, linked into the Opacity input of the brush output node

static ui_node_t *ui_brush_bar_brush_output() {
	ui_node_canvas_t *canvas = g_context->brush->canvas;
	for (i32 i = 0; i < canvas->nodes->length; ++i) {
		if (string_equals(canvas->nodes->buffer[i]->type, "brush_output_node")) {
			return canvas->nodes->buffer[i];
		}
	}
	return NULL;
}

static void ui_brush_bar_set_mask(i32 asset_index) {
	ui_node_canvas_t *canvas = g_context->brush->canvas;
	ui_node_t        *nout   = ui_brush_bar_brush_output();
	if (nout == NULL) {
		return;
	}
	ui_node_t *image = NULL;
	for (i32 i = 0; i < canvas->links->length; ++i) {
		ui_node_link_t *l = canvas->links->buffer[i];
		if (l->to_id == nout->id && l->to_socket == 4) {
			ui_node_t *from = ui_get_node(canvas->nodes, l->from_id);
			if (asset_index >= 0 && from != NULL && string_equals(from->type, "TEX_IMAGE")) {
				image = from; // Reuse the plugged image node
			}
			else {
				for (i32 j = i; j < canvas->links->length - 1; ++j) {
					canvas->links->buffer[j] = canvas->links->buffer[j + 1];
				}
				canvas->links->length--;
			}
			break;
		}
	}
	if (asset_index >= 0) {
		if (image == NULL) {
			image    = nodes_brush_create_node("TEX_IMAGE");
			image->x = nout->x - 220;
			image->y = nout->y + 120;
			ui_node_link_t *l = ALLOC_INIT(ui_node_link_t, {.id = ui_next_link_id(canvas->links), .from_id = image->id, .from_socket = 0, .to_id = nout->id,
			                                                .to_socket = 4});
			any_array_push(canvas->links, l);
		}
		image->buttons->buffer[0]->default_value = f32_array_create_x(asset_index);
	}
	make_material_parse_brush();
	make_material_parse_paint_material(true);
	ui_header_handle->redraws = 2;
}

static void ui_brush_bar_mask_menu() {
	if (ui_icon_button(tr("No Mask"), ICON_CLOSE, UI_ALIGN_LEFT)) {
		ui_brush_bar_set_mask(-1);
	}
	asset_t_array_t *assets = g_project->_->assets;
	if (assets->length == 0) {
		ui_text(tr("No textures imported"), UI_ALIGN_LEFT, 0);
		return;
	}
	for (i32 i = 0; i < assets->length; i += 4) {
		ui_row4();
		for (i32 j = i; j < i + 4; ++j) {
			if (j >= assets->length) {
				ui_end_element();
				continue;
			}
			gpu_texture_t *img = project_get_image(assets->buffer[j]);
			if (img == NULL) {
				ui_end_element();
				continue;
			}
			u32 tint = img == g_context->brush_mask_image ? g_theme->HIGHLIGHT_COL : 0xffffffff;
			if (ui_image(img, tint, -1) == UI_STATE_RELEASED) {
				ui_brush_bar_set_mask(j);
			}
			if (g_ui->is_hovered) {
				ui_tooltip(assets->buffer[j]->name);
			}
		}
	}
}

static void ui_brush_bar_color_menu() {
	ui_text(ui_brush_bar_edit_secondary ? tr("Secondary Color") : tr("Primary Color"), UI_ALIGN_LEFT, g_theme->SEPARATOR_COL);
	ui_color_wheel(&ui_brush_bar_edit_color, &ui_brush_bar_color_state, false, -1, 10 * g_theme->ELEMENT_H * UI_SCALE(), true, NULL, NULL);
	if (ui_item_changed()) {
		if (ui_brush_bar_edit_secondary) {
			ui_brush_bar_secondary = ui_brush_bar_edit_color;
		}
		else {
			ui_brush_bar_set_primary(ui_brush_bar_edit_color);
		}
		ui_header_handle->redraws = 2;
	}
}

static void ui_brush_bar_advanced_menu() {
	ui_brush_bar_advanced = true;
	ui_header_draw_tool_properties();
	ui_brush_bar_advanced = false;
}

// Image widget scaled to the element height, true when released with the position relative to the image
static bool ui_brush_bar_image(gpu_texture_t *img, f32 *rel_x, f32 *rel_y) {
	f32 h  = UI_ELEMENT_H();
	f32 w  = img->width * h / img->height;
	f32 x0 = g_ui->_x;
	f32 y0 = g_ui->_y;
	g_ui->_w = math_floor(w + 4 * UI_SCALE());
	i32 state = ui_image(img, 0xffffffff, h);
	if (state != UI_STATE_RELEASED) {
		return false;
	}
	*rel_x = (g_ui->input_x - g_ui->_window_x - x0) / w;
	*rel_y = (g_ui->input_y - g_ui->_window_y - y0) / h;
	return true;
}

// Left part of the header for painting tools
void ui_brush_bar_draw() {
	i32  _w     = g_ui->_w;
	bool paints = g_context->tool == TOOL_TYPE_BRUSH || g_context->tool == TOOL_TYPE_FILL || g_context->tool == TOOL_TYPE_DECAL ||
	              g_context->tool == TOOL_TYPE_TEXT;
	bool stroke = g_context->tool == TOOL_TYPE_BRUSH || g_context->tool == TOOL_TYPE_ERASER;

	g_ui->_w = math_floor(UI_ELEMENT_H() + 6 * UI_SCALE());
	if (ui_icon_button("", ICON_MENU, UI_ALIGN_CENTER)) {
		ui_menu_draw(&ui_brush_bar_advanced_menu, -1, -1);
	}
	if (g_ui->is_hovered) {
		ui_tooltip(tr("Advanced options"));
	}

	u32 primary = ui_brush_bar_get_primary();
	if (paints) {
		ui_brush_bar_update_colors(primary, ui_brush_bar_secondary);
		bool has_node = ui_brush_bar_color_node() != NULL;
		g_ui->enabled = has_node;
		f32 rx, ry;
		if (ui_brush_bar_image(ui_brush_bar_colors_image, &rx, &ry)) {
			if (rx > 0.62 && ry < 0.34) {
				ui_brush_bar_swap();
			}
			else {
				ui_brush_bar_edit_secondary = rx > 0.55 && ry > 0.62;
				ui_brush_bar_edit_color     = ui_brush_bar_edit_secondary ? ui_brush_bar_secondary : primary;
				ui_menu_draw(&ui_brush_bar_color_menu, -1, -1);
			}
		}
		if (g_ui->is_hovered) {
			any_map_t *vars = any_map_create();
			any_map_set(vars, "key", ui_brush_bar_swap_key());
			ui_tooltip(has_node ? vtr("Primary and secondary color, click to edit, arrow or {key} to swap", vars)
			                    : tr("The base color of this material does not come from a Color node"));
			map_free(vars);
		}
		g_ui->enabled = true;
	}

	if (stroke) {
		gpu_texture_t *mask = g_context->brush_mask_image != NULL ? g_context->brush_mask_image : ui_brush_bar_round();
		g_ui->_w            = math_floor(UI_ELEMENT_H() + 4 * UI_SCALE());
		if (ui_image(mask, 0xffffffff, UI_ELEMENT_H()) == UI_STATE_RELEASED) {
			ui_menu_draw(&ui_brush_bar_mask_menu, -1, -1);
		}
		if (g_ui->is_hovered) {
			ui_tooltip(tr("Brush mask, click to choose an image"));
		}

		ui_brush_bar_update_preview(primary);
		f32 px, py;
		ui_brush_bar_image(ui_brush_bar_preview_image, &px, &py);
	}
	g_ui->_w = _w;
}

// Symmetry toggles aligned to the right of the header
void ui_brush_bar_draw_symmetry() {
	i32  _w      = g_ui->_w;
	f32  bw      = math_floor(26 * UI_SCALE());
	f32  right   = g_ui->_window_w - bw * 3 - 8 * UI_SCALE();
	bool changed = false;
	if (g_ui->_x < right) {
		g_ui->_x = right;
	}
	g_ui->_w            = bw;
	bool *axes[3]       = {&g_context->sym_x, &g_context->sym_y, &g_context->sym_z};
	char *names[3]      = {"X", "Y", "Z"};
	i32   _button_col   = g_theme->BUTTON_COL;
	for (i32 i = 0; i < 3; ++i) {
		g_theme->BUTTON_COL = *axes[i] ? g_theme->HIGHLIGHT_COL : _button_col;
		if (ui_button(names[i], UI_ALIGN_CENTER, "")) {
			*axes[i] = !*axes[i];
			changed  = true;
		}
		if (g_ui->is_hovered) {
			ui_tooltip(string_tmp("%s %s", tr("Symmetry"), names[i]));
		}
	}
	g_theme->BUTTON_COL = _button_col;
	g_ui->_w            = _w;
	if (changed) {
		make_material_parse_paint_material(true);
	}
}

static char *ui_brush_bar_swap_key() {
	char *key = any_map_get(g_keymap, "swap_colors"); // Missing from keymaps saved before
	return key != NULL ? key : "shift+x";
}

// Shortcuts of the brush bar, checked every frame
void ui_brush_bar_update() {
	if (g_ui->is_typing || ui_menu_show) {
		return;
	}
	if (keymap_shortcut(ui_brush_bar_swap_key(), SHORTCUT_TYPE_STARTED)) {
		ui_brush_bar_swap();
	}
}
