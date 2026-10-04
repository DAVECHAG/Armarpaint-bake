
#include "../global.h"

void util_resize_borders() {
	if (ui_base_border_handle != NULL) {
		if (ui_base_border_handle == ui_nodes_hwnd || ui_base_border_handle == ui_view2d_hwnd) {
			if (ui_base_border_started == BORDER_SIDE_LEFT) {
				g_config->layout->buffer[LAYOUT_SIZE_NODES_W] -= math_floor(mouse_movement_x);
				if (g_config->layout->buffer[LAYOUT_SIZE_NODES_W] < 32) {
					g_config->layout->buffer[LAYOUT_SIZE_NODES_W] = 32;
				}
				else if (g_config->layout->buffer[LAYOUT_SIZE_NODES_W] > iron_window_width() * 0.7) {
					g_config->layout->buffer[LAYOUT_SIZE_NODES_W] = math_floor(iron_window_width() * 0.7);
				}
			}
			else { // UINodes / UIView2D ratio
				g_config->layout->buffer[LAYOUT_SIZE_NODES_H] -= math_floor(mouse_movement_y);
				if (g_config->layout->buffer[LAYOUT_SIZE_NODES_H] < 32) {
					g_config->layout->buffer[LAYOUT_SIZE_NODES_H] = 32;
				}
				else if (g_config->layout->buffer[LAYOUT_SIZE_NODES_H] > sys_h() * 0.95) {
					g_config->layout->buffer[LAYOUT_SIZE_NODES_H] = math_floor(sys_h() * 0.95);
				}
			}
		}
		else if (ui_base_border_handle == ui_base_hwnds->buffer[TAB_AREA_STATUS]) {
			i32 my = math_floor(mouse_movement_y);
			if (g_config->layout->buffer[LAYOUT_SIZE_STATUS_H] - my >= ui_statusbar_default_h * g_config->window_scale &&
			    g_config->layout->buffer[LAYOUT_SIZE_STATUS_H] - my < iron_window_height() * 0.7) {
				g_config->layout->buffer[LAYOUT_SIZE_STATUS_H] -= my;
			}
		}
		else { // Sidebar
			if (ui_base_border_started == BORDER_SIDE_LEFT) {
				g_config->layout->buffer[LAYOUT_SIZE_SIDEBAR_W] -= math_floor(mouse_movement_x);
				if (g_config->layout->buffer[LAYOUT_SIZE_SIDEBAR_W] < ui_sidebar_w_mini) {
					g_config->layout->buffer[LAYOUT_SIZE_SIDEBAR_W] = ui_sidebar_w_mini;
				}
				else if (g_config->layout->buffer[LAYOUT_SIZE_SIDEBAR_W] > iron_window_width() - ui_sidebar_w_mini * 2) {
					g_config->layout->buffer[LAYOUT_SIZE_SIDEBAR_W] = iron_window_width() - ui_sidebar_w_mini * 2;
				}

				if (ui_nodes_show || ui_base_show) {
					// Scale down node view if viewport is already at minimal size
					if (g_config->layout->buffer[LAYOUT_SIZE_NODES_W] + g_config->layout->buffer[LAYOUT_SIZE_SIDEBAR_W] + ui_sidebar_w_mini >
					    iron_window_width()) {
						g_config->layout->buffer[LAYOUT_SIZE_NODES_W] =
						    iron_window_width() - g_config->layout->buffer[LAYOUT_SIZE_SIDEBAR_W] - ui_sidebar_w_mini;
					}
				}
			}
			else {
				i32 my = math_floor(mouse_movement_y);
				if (ui_base_border_handle == ui_base_hwnds->buffer[TAB_AREA_SIDEBAR1] && ui_base_border_started == BORDER_SIDE_TOP) {
					if (g_config->layout->buffer[LAYOUT_SIZE_SIDEBAR_H0] + my > 32 && g_config->layout->buffer[LAYOUT_SIZE_SIDEBAR_H1] - my > 32) {
						g_config->layout->buffer[LAYOUT_SIZE_SIDEBAR_H0] += my;
						g_config->layout->buffer[LAYOUT_SIZE_SIDEBAR_H1] -= my;
					}
				}
			}
		}
		base_resize();
	}
}
