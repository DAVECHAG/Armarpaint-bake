
#include "../global.h"

void util_select_update() {
	if (g_context->tool == TOOL_TYPE_SELECT) {
		if (base_ui_enabled && !base_is_dragging && !base_is_resizing) {
			f32  mx          = mouse_view_x();
			f32  my          = mouse_view_y();
			bool in_viewport = context_in_3d_view();

			if (context_is_floating_toolbar() && mx < ui_toolbar_x() + ui_toolbar_w(false)) {
				in_viewport = false;
			}

			if (mouse_started("left") && in_viewport) {
				g_context->select_dragging = true;
				g_context->select_start_x  = mx / (float)sys_w();
				g_context->select_start_y  = my / (float)sys_h();
				g_context->select_x1       = g_context->select_start_x;
				g_context->select_y1       = g_context->select_start_y;
				g_context->select_x2       = g_context->select_start_x;
				g_context->select_y2       = g_context->select_start_y;
			}

			if (g_context->select_dragging && mouse_down("left")) {
				f32 cx               = math_max(0.0, math_min(1.0, mx / (float)sys_w()));
				f32 cy               = math_max(0.0, math_min(1.0, my / (float)sys_h()));
				g_context->select_x1 = math_min(g_context->select_start_x, cx);
				g_context->select_y1 = math_min(g_context->select_start_y, cy);
				g_context->select_x2 = math_max(g_context->select_start_x, cx);
				g_context->select_y2 = math_max(g_context->select_start_y, cy);
			}

			if (g_context->select_dragging && mouse_released("left")) {
				g_context->select_dragging = false;
				f32  dx                    = math_abs(g_context->select_x2 - g_context->select_x1) * sys_w();
				f32  dy                    = math_abs(g_context->select_y2 - g_context->select_y1) * sys_h();
				bool was_active            = g_context->select_active;
				// Drag to enable, click to cancel
				g_context->select_active = dx > 3 && dy > 3;
				if (was_active != g_context->select_active) {
					make_material_parse_paint_material(false);
				}
			}
		}
	}
	else {
		g_context->select_dragging = false;
	}
}
