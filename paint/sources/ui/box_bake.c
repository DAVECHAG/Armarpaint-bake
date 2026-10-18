
#include "../global.h"

// Bake Mesh Maps dialog: bakes a list of maps in sequence and saves each one
// as a png file which is then added to the project textures

typedef struct box_bake_map {
	bake_type_t type;
	char       *name;
	char       *suffix;
	bool        raytraced;
} box_bake_map_t;

static box_bake_map_t box_bake_maps[] = {
    {BAKE_TYPE_NORMAL, "Normal", "normal", false},
    {BAKE_TYPE_HEIGHT, "Height", "height", false},
    {BAKE_TYPE_OCCLUSION, "Ambient Occlusion", "ao", true},
    {BAKE_TYPE_CURVATURE, "Curvature", "curvature", false},
    {BAKE_TYPE_THICKNESS, "Thickness", "thickness", true},
    {BAKE_TYPE_BENT_NORMAL, "Bent Normal", "bent_normal", true},
    {BAKE_TYPE_NORMAL_OBJECT, "World Space Normal", "world_normal", false},
    {BAKE_TYPE_POSITION, "Position", "position", false},
    {BAKE_TYPE_MATERIALID, "Material ID", "material_id", false},
    {BAKE_TYPE_OBJECTID, "Object ID", "object_id", false},
    {BAKE_TYPE_VERTEX_COLOR, "Vertex Color", "vertex_color", false},
    {BAKE_TYPE_TEXCOORD, "TexCoord", "texcoord", false},
    {BAKE_TYPE_CAVITY, "Cavity", "cavity", false},
};
#define BOX_BAKE_MAP_COUNT (sizeof(box_bake_maps) / sizeof(box_bake_maps[0]))

static bool box_bake_enabled[BOX_BAKE_MAP_COUNT] = {true, false, true, true, true, false, false, true, false, false, false, false};

static char *box_bake_folder = NULL;
static char *box_bake_prefix = NULL;
static i32   box_bake_tab    = 0;

// High poly source for Normal and Height, the low poly bakes from itself when disabled
static bool  box_bake_use_high_poly    = false;
static char *box_bake_high_poly_path   = NULL;
static char *box_bake_high_poly_loaded = NULL;
static char *box_bake_high_poly_tried  = NULL;
static f32   box_bake_ray_dist         = 0.02; // Relative to the low poly bounds diagonal
static bool  box_bake_rays_active      = false; // High poly is projected

// Device computing the maps, the cpu bakes every map without raytracing support
#define BOX_BAKE_DEVICE_GPU 0
#define BOX_BAKE_DEVICE_CPU 1
static i32   box_bake_device   = BOX_BAKE_DEVICE_GPU;
static f32   box_bake_cpu_rays = 64;
static f32   box_bake_cavity_range    = 2.0; // Sampling distance in texels
static f32   box_bake_cavity_contrast = 1.0;
static bool  box_bake_cpu_ready = false;

// 3D preview of the low poly, high poly and cage
#define BOX_BAKE_PREVIEW_W 620
#define BOX_BAKE_PREVIEW_H 470
static gpu_texture_t *box_bake_preview_image     = NULL;
static gpu_texture_t *box_bake_preview_old       = NULL;
static u8            *box_bake_preview_pixels    = NULL;
static bool           box_bake_preview_dirty     = true;
static bool           box_bake_preview_low_ready = false;
static f32            box_bake_preview_yaw       = 0.8;
static f32            box_bake_preview_pitch     = 0.35;
static f32            box_bake_preview_zoom      = 1.0;
static bool           box_bake_show_low          = true;
static bool           box_bake_show_high         = false;
static bool           box_bake_show_cage         = true;
static bool           box_bake_show_miss         = true;
static bool           box_bake_average_normals   = true;

// Bake queue state
static bool             box_bake_running = false;
static i32_array_t     *box_bake_queue   = NULL;
static i32              box_bake_queue_i = 0;
static render_target_t *box_bake_rt      = NULL;
static render_target_t *box_bake_layer_rt;
static tool_type_t      box_bake_tool;
static i32              box_bake_bits;
static i32              box_bake_done_count;

static bool box_bake_use_cpu() {
	return box_bake_device == BOX_BAKE_DEVICE_CPU || !gpu_raytrace_supported();
}

static bool box_bake_map_available(i32 i) {
	return box_bake_use_cpu() || !box_bake_maps[i].raytraced || gpu_raytrace_supported();
}

// High poly Normal and Height are always projected on the cpu
static bool box_bake_map_on_cpu(box_bake_map_t *map) {
	return box_bake_use_cpu() || map->type == BAKE_TYPE_CAVITY || (box_bake_rays_active && (map->type == BAKE_TYPE_NORMAL || map->type == BAKE_TYPE_HEIGHT));
}

static bool box_bake_map_rgba128(bake_type_t type) {
	return type == BAKE_TYPE_NORMAL || type == BAKE_TYPE_HEIGHT || type == BAKE_TYPE_DERIVATIVE;
}

static char *box_bake_default_prefix() {
	if (g_context->paint_object != NULL && g_context->paint_object->base->name != NULL) {
		return g_context->paint_object->base->name;
	}
	return "bake";
}

// Baked maps are associated to the mesh by file name: <mesh name>_<map>.png
gpu_texture_t *box_bake_mesh_map_image(char *suffix) {
	asset_t_array_t *assets = g_project->_->assets;
	if (assets == NULL || assets->length == 0) {
		return NULL;
	}
	for (i32 k = -2; k < g_project->_->paint_objects->length; ++k) {
		char *name = k == -2   ? box_bake_prefix
		             : k == -1 ? (g_context->paint_object != NULL ? g_context->paint_object->base->name : NULL)
		                       : g_project->_->paint_objects->buffer[k]->base->name;
		if (name == NULL || string_equals(name, "")) {
			continue;
		}
		char *file_name = string_tmp("%s_%s.png", name, suffix);
		for (i32 i = 0; i < assets->length; ++i) {
			asset_t *a = assets->buffer[i];
			if (a->image != NULL && string_equals(a->name, file_name)) {
				return a->image;
			}
		}
	}
	return NULL;
}

// True when a visible fill layer uses a material reading this image, the layer then
// already carries the baked map and the viewport must not apply it a second time
static bool box_bake_image_in_fill_layer(gpu_texture_t *image) {
	slot_layer_t_array_t *layers = g_project->_->layers;
	asset_t_array_t      *assets = g_project->_->assets;
	for (i32 i = 0; i < layers->length; ++i) {
		slot_layer_t *l = layers->buffer[i];
		if (!l->visible || l->fill_material == NULL) {
			continue;
		}
		ui_node_canvas_t *canvas = l->fill_material->canvas;
		for (i32 j = 0; j < canvas->nodes->length; ++j) {
			ui_node_t *n = canvas->nodes->buffer[j];
			if (!string_equals(n->type, "TEX_IMAGE")) {
				continue;
			}
			i32 index = (i32)n->buttons->buffer[0]->default_value->buffer[0];
			if (index >= 0 && index < assets->length && assets->buffer[index]->image == image) {
				return true;
			}
		}
	}
	return false;
}

// Baked map to apply in the viewport, NULL when missing or already used by a fill layer
gpu_texture_t *box_bake_mesh_map_viewport(char *suffix) {
	gpu_texture_t *image = box_bake_mesh_map_image(suffix);
	if (image == NULL || box_bake_image_in_fill_layer(image)) {
		return NULL;
	}
	return image;
}

// Rebuild the viewport shader when the baked maps it applies change, for example
// when a fill layer starts or stops using them
void box_bake_watch_viewport_maps(void *_) {
	static gpu_texture_t *last_ao  = NULL;
	static gpu_texture_t *last_nor = NULL;
	if (g_project == NULL || g_project->_->layers == NULL) {
		return;
	}
	gpu_texture_t *ao  = box_bake_mesh_map_viewport("ao");
	gpu_texture_t *nor = box_bake_mesh_map_viewport("normal");
	if (ao != last_ao || nor != last_nor) {
		last_ao  = ao;
		last_nor = nor;
		make_material_parse_mesh_material();
		g_context->ddirty = 2;
	}
}

// The high poly only lives in the ray tracer, released when the dialog closes
static void box_bake_unload_high_poly() {
	box_bake_high_poly_loaded = NULL;
	box_bake_high_poly_tried  = NULL;
	bake_rays_set_high_raw(NULL);
	box_bake_preview_dirty = true;
}

static void box_bake_free_raw_array(void *a) {
	if (a != NULL) {
		free(((i16_array_t *)a)->buffer);
		free(a);
	}
}

static raw_mesh_t_array_t *box_bake_parse_high_poly(char *path) {
	raw_mesh_t_array_t *parts = any_array_create_from_raw((void *[]){}, 0);
	char               *p     = to_lower_case(path);
	bool                is_obj = ends_with(p, ".obj");
	free(p);

	if (is_obj) {
		buffer_t *b = data_get_blob(path);
		if (b == NULL) {
			return parts;
		}
		raw_mesh_t *part = obj_parse(b, 'o', 0, false);
		any_array_push(parts, part);
		while (part->has_next) {
			part = obj_parse(b, 'o', part->pos, false);
			if (part->inda->length > 0) {
				any_array_push(parts, part);
			}
		}
		data_delete_blob(path);
		return parts;
	}

	if (!path_is_mesh(path) && !context_enable_import_plugin(path)) {
		return parts;
	}
	char *ext                           = string_tmp("%s", path + string_last_index_of(path, ".") + 1);
	raw_mesh_t *(*importer)(char *path) = any_map_get(import_mesh_importers, ext);
	if (importer == NULL) {
		return parts;
	}
	raw_mesh_t *mesh = importer(path);
	while (mesh != NULL) {
		any_array_push(parts, mesh);
		mesh = mesh->has_next ? importer(path) : NULL;
	}
	return parts;
}

// Read the high poly file straight into the ray tracer, it is never added to the scene
static bool box_bake_load_high_poly() {
	if (box_bake_high_poly_loaded != NULL && string_equals(box_bake_high_poly_loaded, box_bake_high_poly_path)) {
		return true;
	}
	box_bake_unload_high_poly();

	raw_mesh_t_array_t *parts = box_bake_parse_high_poly(box_bake_high_poly_path);
	bool                ok    = bake_rays_set_high_raw(parts);

	// The importers malloc the raw meshes, nothing else references them
	for (i32 i = 0; i < parts->length; ++i) {
		raw_mesh_t *m = parts->buffer[i];
		box_bake_free_raw_array(m->posa);
		box_bake_free_raw_array(m->nora);
		box_bake_free_raw_array(m->texa);
		box_bake_free_raw_array(m->texa1);
		box_bake_free_raw_array(m->cola);
		box_bake_free_raw_array(m->inda);
		free(m);
	}
	parts->length = 0;

	if (!ok) {
		console_error(string("%s: %s", tr("Failed to load high poly mesh"), box_bake_high_poly_path));
		return false;
	}
	box_bake_high_poly_loaded = string_copy(box_bake_high_poly_path);
	return true;
}

static bool box_bake_any_selected(bake_type_t a, bake_type_t b);

static void box_bake_clear(gpu_texture_t *t) {
	draw_begin(t, true, 0xff000000);
	draw_end();
}

static void box_bake_write_and_import(box_bake_map_t *map, buffer_t *pixels, i32 w, i32 h);

static void box_bake_save_result(box_bake_map_t *map) {
	// Copy to an 8bit target for png output
	gpu_texture_t *src  = box_bake_rt->_image;
	gpu_texture_t *temp = gpu_create_render_target(src->width, src->height, GPU_TEXTURE_FORMAT_RGBA32);
	draw_begin(temp, false, 0);
	draw_scaled_image(src, 0, 0, src->width, src->height);
	draw_end();

	buffer_t *pixels = gpu_get_texture_pixels(temp);
#ifdef IRON_BGRA
	buffer_bgra_swap(pixels);
#endif
	box_bake_write_and_import(map, pixels, temp->width, temp->height);
	gpu_delete_texture(temp);
}

static void box_bake_write_and_import(box_bake_map_t *map, buffer_t *pixels, i32 w, i32 h) {
	char *file = string("%s%s%s_%s.png", box_bake_folder, PATH_SEP, box_bake_prefix, map->suffix);
	iron_write_png(file, pixels, w, h, 0);

	// Add to project textures, or refresh the existing one
	asset_t *existing = NULL;
	for (i32 i = 0; i < g_project->_->assets->length; ++i) {
		asset_t *a = g_project->_->assets->buffer[i];
		if (string_equals(a->file, file)) {
			existing = a;
			break;
		}
	}
	if (existing != NULL) {
		project_reimport_texture(existing);
	}
	else {
		import_texture_run(file, false);
	}
	box_bake_done_count++;
}

static void box_bake_restore_layer() {
	any_map_set(render_path_render_targets, string("texpaint%d", g_context->layer->id), box_bake_layer_rt);
	g_context->layer->texpaint = bake_texture_node_texpaint;
	g_context->tool            = box_bake_tool;
	bake_texture_node_baking   = false;
}

// Baked maps plugged into the material output, socket indices of OUTPUT_MATERIAL_PBR
typedef struct box_bake_socket {
	char *suffix;
	i32   socket;
} box_bake_socket_t;

static box_bake_socket_t box_bake_sockets[] = {
    {"ao", 2},
    {"normal", 5},
    {"height", 7},
};

static bool box_bake_has_asset(char *file_name) {
	for (i32 i = 0; i < g_project->_->assets->length; ++i) {
		if (string_equals(g_project->_->assets->buffer[i]->name, file_name)) {
			return true;
		}
	}
	return false;
}

// Create or update the "<name> Baked" material with the baked maps linked to the output
static void box_bake_update_material() {
	char *name = string("%s Baked", box_bake_prefix);
	bool  any  = false;
	for (i32 k = 0; k < 3; ++k) {
		any |= box_bake_has_asset(string_tmp("%s_%s.png", box_bake_prefix, box_bake_sockets[k].suffix));
	}
	if (!any) {
		return;
	}

	slot_material_t *mat = NULL;
	for (i32 i = 0; i < g_project->_->materials->length; ++i) {
		slot_material_t *m = g_project->_->materials->buffer[i];
		if (string_equals(m->canvas->name, name)) {
			mat = m;
			break;
		}
	}
	bool created = mat == NULL;
	if (created) {
		mat = slot_material_create(g_project->_->materials->buffer[0]->data, NULL);
		any_array_push(g_project->_->materials, mat);
		mat->canvas->name = string_copy(slot_material_unique_name(mat->canvas, name));
	}
	ui_node_canvas_t *canvas = mat->canvas;
	ui_node_t        *nout   = NULL;
	for (i32 i = 0; i < canvas->nodes->length; ++i) {
		ui_node_t *n = canvas->nodes->buffer[i];
		if (string_equals(n->type, "OUTPUT_MATERIAL_PBR")) {
			nout = n;
			break;
		}
	}
	if (nout == NULL) {
		return;
	}

	// Image nodes go left of the existing ones
	f32 x = nout->x;
	for (i32 i = 0; i < canvas->nodes->length; ++i) {
		ui_node_t *n = canvas->nodes->buffer[i];
		if (!string_equals(n->type, "TEX_IMAGE")) {
			x = math_min(x, n->x);
		}
	}
	x -= 260;

	for (i32 k = 0; k < 3; ++k) {
		char *file = string_tmp("%s_%s.png", box_bake_prefix, box_bake_sockets[k].suffix);
		if (!box_bake_has_asset(file)) {
			continue;
		}
		// Replace the image node already plugged into this socket, other nodes are kept
		for (i32 i = 0; i < canvas->links->length; ++i) {
			ui_node_link_t *l = canvas->links->buffer[i];
			if (l->to_id == nout->id && l->to_socket == box_bake_sockets[k].socket) {
				ui_node_t *from = ui_get_node(canvas->nodes, l->from_id);
				if (from != NULL && string_equals(from->type, "TEX_IMAGE")) {
					ui_remove_node(from, canvas);
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
		ui_node_t *n = ui_nodes_make_node(nodes_material_get_node_t("TEX_IMAGE"), mat->nodes, canvas);
		any_array_push(canvas->nodes, n);
		n->buttons->buffer[0]->default_value = f32_array_create_x(base_get_asset_index(file));
		n->x                                 = x;
		n->y                                 = 100 + 180 * k;
		ui_node_link_t *l                    = ALLOC_INIT(ui_node_link_t, {.id          = ui_next_link_id(canvas->links),
		                                                                   .from_id     = n->id,
		                                                                   .from_socket = 0,
		                                                                   .to_id       = nout->id,
		                                                                   .to_socket   = box_bake_sockets[k].socket});
		any_array_push(canvas->links, l);
	}

	g_context->material = mat;
	make_material_parse_paint_material(true);
	util_render_make_material_preview();
	ui_dock_redraw(1);
	if (created) {
		history_new_material();
	}
	any_map_t *vars = any_map_create();
	any_map_set(vars, "name", mat->canvas->name);
	console_info(vtr("Baked maps linked in material {name}", vars));
	map_free(vars);
}

static void box_bake_finish() {
	box_bake_running     = false;
	box_bake_rays_active = false;
	box_bake_cpu_ready   = false;
	bake_rays_end_images();
	box_bake_preview_low_ready = false; // Geometry was reread by the bake
	make_material_parse_paint_material(true);
	make_material_parse_mesh_material();
	g_context->ddirty = 2;
	layers_update_fill_layers();
	ui_view2d_hwnd->redraws = 2;
	ui_box_hwnd->redraws    = 2;
	ui_dock_redraw(TAB_AREA_STATUS);

	if (box_bake_done_count > 0) {
		box_bake_update_material();
	}

	any_map_t *vars = any_map_create();
	any_map_set(vars, "count", i32_to_string(box_bake_done_count));
	console_info(vtr("Baked {count} maps", vars));
	map_free(vars);
}

static void box_bake_start_next(void *_);
static void box_bake_finish();
static void box_bake_advance();

static void box_bake_check_result(void *_) {
	ui_box_hwnd->redraws = 2;

	if (g_context->pdirty > 0 || render_path_paint_baking) {
		return;
	}
	sys_remove_update(box_bake_check_result);

	box_bake_map_t *map     = &box_bake_maps[box_bake_queue->buffer[box_bake_queue_i]];
	bool            rgba128 = box_bake_map_rgba128(map->type);

	box_bake_restore_layer();
	if (rgba128) {
		base_bits = box_bake_bits;
		layers_set_bits();
	}

	if (box_bake_running) {
		box_bake_save_result(map);
	}
	box_bake_advance();
}

static void box_bake_advance() {
	box_bake_queue_i++;
	if (box_bake_running && box_bake_queue_i < box_bake_queue->length) {
		sys_notify_on_next_frame(&box_bake_start_next, NULL);
	}
	else {
		box_bake_finish();
	}
}

static void box_bake_save_rays_map(box_bake_map_t *map) {
	i32      w      = config_get_texture_res_x();
	i32      h      = config_get_texture_res_y();
	buffer_t pixels = {.buffer = bake_rays_pixels(), .length = w * h * 4};
	box_bake_write_and_import(map, &pixels, w, h);
	box_bake_advance();
}

static void box_bake_rays_update(void *_) {
	ui_box_hwnd->redraws = 2;
	if (!box_bake_running) {
		bake_rays_cancel();
		sys_remove_update(box_bake_rays_update);
		box_bake_finish();
		return;
	}
	if (!bake_rays_step()) {
		return;
	}
	sys_remove_update(box_bake_rays_update);
	box_bake_save_rays_map(&box_bake_maps[box_bake_queue->buffer[box_bake_queue_i]]);
}

static void box_bake_start_next(void *_) {
	box_bake_map_t *map = &box_bake_maps[box_bake_queue->buffer[box_bake_queue_i]];
	box_bake_tool       = g_context->tool;

	// Cpu maps run on worker threads, polled every update
	if (box_bake_map_on_cpu(map)) {
		if (!box_bake_cpu_ready) {
			box_bake_cpu_ready = bake_rays_begin(g_project->_->paint_objects, box_bake_rays_active,
			                                     config_get_texture_res_x(), config_get_texture_res_y(), box_bake_ray_dist);
			if (!box_bake_cpu_ready) {
				console_error(tr("Unable to bake on the CPU, check that the low poly has a UV map"));
				box_bake_advance();
				return;
			}
		}
		bake_rays_set_cavity(box_bake_cavity_range, box_bake_cavity_contrast);
		bake_rays_start(map->type, box_bake_rays_active, (i32)box_bake_cpu_rays);
		sys_notify_on_update(box_bake_rays_update, NULL);
		return;
	}

	// Dedicated render target, recreated when resolution or format changes
	bool rgba128 = box_bake_map_rgba128(map->type);
	i32  res_x   = config_get_texture_res_x();
	i32  res_y   = config_get_texture_res_y();
	if (box_bake_rt == NULL) {
		box_bake_rt         = render_target_create();
		box_bake_rt->name   = "box_bake";
		box_bake_rt->width  = res_x;
		box_bake_rt->height = res_y;
		box_bake_rt->format = "RGBA32";
		render_path_create_render_target(box_bake_rt);
	}
	gpu_delete_texture(box_bake_rt->_image);
	box_bake_rt->width  = res_x;
	box_bake_rt->height = res_y;
	box_bake_rt->format = rgba128 ? "RGBA128" : "RGBA32";
	box_bake_rt->_image = gpu_create_render_target(res_x, res_y, rgba128 ? GPU_TEXTURE_FORMAT_RGBA128 : GPU_TEXTURE_FORMAT_RGBA32);

	// Bake in lit mode for now
	if (g_context->viewport_mode == VIEWPORT_MODE_PATH_TRACE) {
		g_context->viewport_mode = VIEWPORT_MODE_LIT;
	}

	g_context->tool      = TOOL_TYPE_BAKE;
	g_context->bake_type = map->type;

	if (rgba128) {
		// Use RGBA128 for high poly to low poly baking to prevent artifacts
		box_bake_bits = base_bits;
		base_bits     = TEXTURE_BITS_BITS32;
		layers_set_bits();
		history_push_undo = true; // Requires 2 steps
	}

	// Render into the bake target in place of the current layer
	i32 lid           = g_context->layer->id;
	box_bake_layer_rt = any_map_get(render_path_render_targets, string_tmp("texpaint%d", lid));
	any_map_set(render_path_render_targets, string("texpaint%d", lid), box_bake_rt);
	bake_texture_node_texpaint = g_context->layer->texpaint;

	g_context->layer->texpaint               = box_bake_rt->_image;
	g_context->pdirty                        = map->raytraced ? g_context->bake_samples : 1;
	g_context->rtdirty                       = 1;
	render_path_raytrace_bake_current_sample = 0;
	render_path_raytrace_frame               = 0;
	bake_texture_node_baking                 = true;
	make_material_parse_paint_material(false);
	sys_notify_on_next_frame(box_bake_clear, box_bake_rt->_image);
	sys_notify_on_update(box_bake_check_result, NULL);
}

static void box_bake_run() {
	if (box_bake_queue == NULL) {
		box_bake_queue = i32_array_create(0);
	}
	box_bake_queue->length = 0;
	for (i32 i = 0; i < (i32)BOX_BAKE_MAP_COUNT; ++i) {
		if (box_bake_enabled[i] && box_bake_map_available(i)) {
			i32_array_push(box_bake_queue, i);
		}
	}
	if (box_bake_queue->length == 0) {
		console_error(tr("No maps selected"));
		return;
	}
	if (string_equals(box_bake_prefix, "")) {
		box_bake_prefix = string_copy(box_bake_default_prefix());
	}

	// Maps bake from the high poly file, or from the low poly itself.
	// On the gpu only Normal and Height use the high poly.
	box_bake_rays_active      = false;
	box_bake_cpu_ready        = false;
	g_context->bake_high_poly = array_index_of(g_project->_->paint_objects, g_context->paint_object);
	if (g_context->bake_high_poly < 0) {
		g_context->bake_high_poly = 0;
	}
	if (box_bake_use_high_poly && (box_bake_use_cpu() || box_bake_any_selected(BAKE_TYPE_NORMAL, BAKE_TYPE_HEIGHT) || box_bake_any_selected(BAKE_TYPE_CAVITY, BAKE_TYPE_CAVITY))) {
		if (string_equals(box_bake_high_poly_path, "")) {
			console_error(tr("Select a high poly mesh file or disable High Poly"));
			return;
		}
		if (!box_bake_load_high_poly()) {
			return;
		}
		box_bake_rays_active = true;
	}

	box_bake_running    = true;
	box_bake_tab        = 0;
	box_bake_queue_i    = 0;
	box_bake_done_count = 0;
	sys_notify_on_next_frame(&box_bake_start_next, NULL);
}

static void box_bake_folder_picked(char *path) {
	box_bake_folder = string_copy(path);
}

static void box_bake_folder_picked_and_run(char *path) {
	box_bake_folder_picked(path);
	box_bake_run();
}

static bool box_bake_any_selected(bake_type_t a, bake_type_t b) {
	for (i32 i = 0; i < (i32)BOX_BAKE_MAP_COUNT; ++i) {
		if (box_bake_enabled[i] && box_bake_map_available(i) && (box_bake_maps[i].type == a || box_bake_maps[i].type == b)) {
			return true;
		}
	}
	return false;
}

// Folder picker reuses the save dialog, keep the project file name intact
static void box_bake_pick_folder(void (*done)(char *)) {
	char *filename = ui_files_filename;
	ui_files_show("png", true, false, done);
	ui_files_filename = filename;
}

static void box_bake_draw_progress() {
	box_bake_map_t *map      = &box_bake_maps[box_bake_queue->buffer[box_bake_queue_i]];
	f32             progress = 0.0;
	if (box_bake_map_on_cpu(map)) {
		progress = bake_rays_progress();
	}
	else if (map->raytraced) {
		progress = render_path_raytrace_bake_current_sample / (float)g_context->bake_samples;
		if (progress > 1.0) {
			progress = 1.0;
		}
	}
	char *label = string_tmp("%s %d/%d: %s %d%%", tr("Baking"), box_bake_queue_i + 1, box_bake_queue->length, tr(map->name), (i32)(progress * 100));
	ui_text(label, UI_ALIGN_LEFT, 0);

	f32 total = (box_bake_queue_i + progress) / (float)box_bake_queue->length;
	i32 _BUTTON_COL     = g_theme->BUTTON_COL;
	g_theme->BUTTON_COL = g_theme->HIGHLIGHT_COL;
	ui_slider(&total, "", 0.0, 1.0, true, 100, false, UI_ALIGN_CENTER, false);
	g_theme->BUTTON_COL = _BUTTON_COL;

	if (ui_icon_button(tr("Stop"), ICON_STOP, UI_ALIGN_CENTER)) {
		box_bake_running  = false;
		g_context->pdirty = 0;
	}
}

static void box_bake_draw_maps() {
	ui_text(tr("Maps"), UI_ALIGN_LEFT, g_theme->SEPARATOR_COL);
	for (i32 i = 0; i < (i32)BOX_BAKE_MAP_COUNT; i += 2) {
		ui_row2();
		for (i32 j = i; j < i + 2 && j < (i32)BOX_BAKE_MAP_COUNT; ++j) {
			g_ui->enabled = box_bake_map_available(j);
			ui_check(&box_bake_enabled[j], tr(box_bake_maps[j].name), "");
			if (g_ui->is_hovered && !g_ui->enabled) {
				ui_tooltip(tr("Requires raytracing support, or set Device to CPU in Settings"));
			}
			g_ui->enabled = true;
		}
		if (i + 1 >= (i32)BOX_BAKE_MAP_COUNT) {
			ui_end_element(); // Odd count, complete the row
		}
	}

	ui_row2();
	if (ui_button(tr("Select All"), UI_ALIGN_CENTER, "")) {
		for (i32 i = 0; i < (i32)BOX_BAKE_MAP_COUNT; ++i) {
			box_bake_enabled[i] = true;
		}
	}
	if (ui_button(tr("Select None"), UI_ALIGN_CENTER, "")) {
		for (i32 i = 0; i < (i32)BOX_BAKE_MAP_COUNT; ++i) {
			box_bake_enabled[i] = false;
		}
	}
}

static void box_bake_draw_device() {
	ui_text(tr("Device"), UI_ALIGN_LEFT, g_theme->SEPARATOR_COL);
	string_array_t *device_combo = any_array_create_from_raw_tmp(
	    (void *[]){
	        tr("GPU"),
	        tr("CPU"),
	    },
	    2);
	bool rt = gpu_raytrace_supported();
	if (!rt) {
		box_bake_device = BOX_BAKE_DEVICE_CPU;
	}
	g_ui->enabled = rt;
	ui_combo(&box_bake_device, device_combo, tr("Bake on"), true, UI_ALIGN_LEFT, true);
	g_ui->enabled = true;
	if (g_ui->is_hovered) {
		ui_tooltip(tr("GPU: fast, Ambient Occlusion, Thickness and Bent Normal need raytracing support. CPU: every map, slower, works on any computer"));
	}
	if (!rt) {
		ui_text(tr("No raytracing support on this GPU, maps are baked on the CPU"), UI_ALIGN_LEFT, 0);
	}
}

static void box_bake_draw_common() {
	ui_text(tr("Common"), UI_ALIGN_LEFT, g_theme->SEPARATOR_COL);

	string_array_t *base_res_combo = any_array_create_from_raw_tmp(
	    (void *[]){
	        "2048",
	        "4096",
	        "8192",
	        "16384",
	        tr("Custom"),
	    },
	    5);
	ui_combo(&base_res, base_res_combo, tr("Resolution"), true, UI_ALIGN_LEFT, true);
	if (ui_item_changed()) {
		config_set_texture_res(base_res);
		layers_on_resized();
	}

	if (box_bake_any_selected(BAKE_TYPE_NORMAL_OBJECT, BAKE_TYPE_POSITION) || box_bake_any_selected(BAKE_TYPE_BENT_NORMAL, BAKE_TYPE_BENT_NORMAL)) {
		string_array_t *up_axis_combo = any_array_create_from_raw_tmp(
		    (void *[]){
		        tr("Z"),
		        tr("Y"),
		    },
		    2);
		ui_combo((int *)&g_context->bake_up_axis, up_axis_combo, tr("Up Axis"), true, UI_ALIGN_LEFT, true);
	}

	if (box_bake_any_selected(BAKE_TYPE_OCCLUSION, BAKE_TYPE_CURVATURE)) {
		string_array_t *axis_combo = any_array_create_from_raw_tmp(
		    (void *[]){
		        tr("XYZ"),
		        tr("X"),
		        tr("Y"),
		        tr("Z"),
		        tr("-X"),
		        tr("-Y"),
		        tr("-Z"),
		    },
		    7);
		ui_combo((int *)&g_context->bake_axis, axis_combo, tr("Axis"), true, UI_ALIGN_LEFT, true);
	}
}

static void box_bake_draw_settings() {
	if (box_bake_any_selected(BAKE_TYPE_OCCLUSION, BAKE_TYPE_THICKNESS) || box_bake_any_selected(BAKE_TYPE_BENT_NORMAL, BAKE_TYPE_BENT_NORMAL)) {
		ui_text(tr("Raytracing"), UI_ALIGN_LEFT, g_theme->SEPARATOR_COL);
		if (box_bake_use_cpu()) {
			ui_slider(&box_bake_cpu_rays, tr("Rays"), 4, 512, true, 1, true, UI_ALIGN_RIGHT, true);
			if (g_ui->is_hovered) {
				ui_tooltip(tr("Rays per texel, more is smoother and slower"));
			}
			box_bake_cpu_rays = math_floor(box_bake_cpu_rays);
		}
		else {
			f32 samples = g_context->bake_samples;
			ui_set_next_id((ui_id_t)&g_context->bake_samples);
			g_context->bake_samples = math_floor(ui_slider(&samples, tr("Samples"), 1, 512, true, 1, true, UI_ALIGN_RIGHT, true));
		}
	}

	if (box_bake_any_selected(BAKE_TYPE_CAVITY, BAKE_TYPE_CAVITY)) {
		ui_text(tr("Cavity"), UI_ALIGN_LEFT, g_theme->SEPARATOR_COL);
		ui_slider(&box_bake_cavity_range, tr("Detail Size"), 1.0, 16.0, true, 10.0, true, UI_ALIGN_RIGHT, true);
		if (g_ui->is_hovered) {
			ui_tooltip(tr("Size of the details in texels: small values for scales and pores, large values for folds"));
		}
		ui_slider(&box_bake_cavity_contrast, tr("Contrast"), 0.5, 6.0, true, 100.0, true, UI_ALIGN_RIGHT, true);
		if (g_ui->is_hovered) {
			ui_tooltip(tr("Higher values darken the cavities more, useful as a mask"));
		}
	}

	if (box_bake_any_selected(BAKE_TYPE_OCCLUSION, BAKE_TYPE_OCCLUSION)) {
		ui_text(tr("Ambient Occlusion"), UI_ALIGN_LEFT, g_theme->SEPARATOR_COL);
		ui_slider(&g_context->bake_ao_strength, tr("Strength"), 0.0, 2.0, true, 100.0, true, UI_ALIGN_RIGHT, true);
		ui_slider(&g_context->bake_ao_radius, tr("Radius"), 0.0, 2.0, true, 100.0, true, UI_ALIGN_RIGHT, true);
		ui_slider(&g_context->bake_ao_offset, tr("Offset"), 0.0, 2.0, true, 100.0, true, UI_ALIGN_RIGHT, true);
	}

	if (box_bake_any_selected(BAKE_TYPE_CURVATURE, BAKE_TYPE_CURVATURE)) {
		ui_text(tr("Curvature"), UI_ALIGN_LEFT, g_theme->SEPARATOR_COL);
		ui_slider(&g_context->bake_curv_strength, tr("Strength"), 0.0, 2.0, true, 100.0, true, UI_ALIGN_RIGHT, true);
		ui_slider(&g_context->bake_curv_radius, tr("Radius"), 0.0, 2.0, true, 100.0, true, UI_ALIGN_RIGHT, true);
		ui_slider(&g_context->bake_curv_offset, tr("Offset"), -2.0, 2.0, true, 100.0, true, UI_ALIGN_RIGHT, true);
		f32 smooth = g_context->bake_curv_smooth;
		ui_set_next_id((ui_id_t)&g_context->bake_curv_smooth);
		g_context->bake_curv_smooth = math_floor(ui_slider(&smooth, tr("Smooth"), 0, 5, false, 1, true, UI_ALIGN_RIGHT, true));
	}
}

static void box_bake_high_poly_picked(char *path) {
	box_bake_high_poly_path = string_copy(path);
	box_bake_use_high_poly  = true;
}

static void box_bake_draw_high_poly() {
	ui_text(tr("High Poly"), UI_ALIGN_LEFT, g_theme->SEPARATOR_COL);
	ui_check(&box_bake_use_high_poly, tr("Bake from High Poly mesh"), "");
	if (g_ui->is_hovered) {
		ui_tooltip(tr("Maps are projected from this mesh with rays along the low poly normals: Normal and Height on the GPU, every map on the CPU. Disabled: bake from the low poly itself"));
	}
	if (!box_bake_use_high_poly) {
		return;
	}
	f32_array_t *row = f32_array_create_from_raw((f32[]){0.8, 0.2}, 2);
	ui_row(row);
	char *path = string_equals(box_bake_high_poly_path, "") ? tr("No file selected") : box_bake_high_poly_path;
	ui_text(path, UI_ALIGN_LEFT, 0);
	if (ui_button("...", UI_ALIGN_CENTER, "")) {
		char *filename = ui_files_filename;
		ui_files_show("obj,fbx,gltf,glb", false, false, &box_bake_high_poly_picked);
		ui_files_filename = filename;
	}
	bool loaded = box_bake_high_poly_loaded != NULL && string_equals(box_bake_high_poly_loaded, box_bake_high_poly_path);
	if (loaded && bake_rays_high_count() > 0) {
		any_map_t *vars = any_map_create();
		any_map_set(vars, "count", i32_to_string(bake_rays_high_count()));
		ui_text(vtr("{count} triangles", vars), UI_ALIGN_LEFT, 0);
		map_free(vars);
		f32 fit = bake_rays_high_fit();
		if (fit < 0.5 || fit > 2.0) {
			ui_text(fit == 0.0 ? tr("Warning: the high poly does not overlap the low poly, check its position and scale at export")
			                   : tr("Warning: the high poly and low poly sizes differ a lot, check the scale at export"),
			        UI_ALIGN_LEFT, 0);
		}
	}
	ui_check(&box_bake_average_normals, tr("Average Normals"), "");
	if (g_ui->is_hovered) {
		ui_tooltip(tr("Cast rays along normals averaged over hard edges, keeps the cage closed"));
	}
	if (ui_item_changed()) {
		bake_rays_set_average(box_bake_average_normals);
		box_bake_preview_dirty = true;
	}
}

static void box_bake_draw_output() {
	ui_text(tr("Output"), UI_ALIGN_LEFT, g_theme->SEPARATOR_COL);
	ui_text_input(&box_bake_prefix, tr("Name"), UI_ALIGN_LEFT, true, false);
	if (g_ui->is_hovered) {
		ui_tooltip(tr("Files are saved as name_map.png"));
	}

	f32_array_t *row = f32_array_create_from_raw((f32[]){0.8, 0.2}, 2);
	ui_row(row);
	char *folder = string_equals(box_bake_folder, "") ? tr("No folder selected") : box_bake_folder;
	ui_text(folder, UI_ALIGN_LEFT, 0);
	if (ui_button("...", UI_ALIGN_CENTER, "")) {
		box_bake_pick_folder(&box_bake_folder_picked);
	}
}

static bool box_bake_preview_fast = false; // Cheaper render while orbiting

static void box_bake_preview_render() {
	if (box_bake_preview_pixels == NULL) {
		box_bake_preview_pixels = malloc(BOX_BAKE_PREVIEW_W * BOX_BAKE_PREVIEW_H * 4);
	}
	bake_rays_set_dist(box_bake_ray_dist);
	bake_rays_preview(box_bake_preview_pixels, BOX_BAKE_PREVIEW_W, BOX_BAKE_PREVIEW_H, box_bake_preview_yaw, box_bake_preview_pitch, box_bake_preview_zoom,
	                  box_bake_show_low, box_bake_show_high, box_bake_show_cage, box_bake_show_miss, box_bake_preview_fast);
	buffer_t b = {.buffer = box_bake_preview_pixels, .length = BOX_BAKE_PREVIEW_W * BOX_BAKE_PREVIEW_H * 4};
	// The previous image may still be referenced by this frame, release it one update later
	if (box_bake_preview_old != NULL) {
		gpu_delete_texture(box_bake_preview_old);
	}
	box_bake_preview_old   = box_bake_preview_image;
	box_bake_preview_image = gpu_create_texture_from_bytes(&b, BOX_BAKE_PREVIEW_W, BOX_BAKE_PREVIEW_H, GPU_TEXTURE_FORMAT_RGBA32);
	box_bake_preview_dirty = false;
}

static void box_bake_draw_preview() {
	// Geometry for the preview, the high poly is loaded once a file is picked
	if (!box_bake_preview_low_ready) {
		bake_rays_set_average(box_bake_average_normals);
		bake_rays_set_low(g_project->_->paint_objects);
		box_bake_preview_low_ready = true;
		box_bake_preview_dirty     = true;
	}
	bool has_high = box_bake_use_high_poly && !string_equals(box_bake_high_poly_path, "");
	if (has_high && (box_bake_show_high || box_bake_show_miss)) {
		bool loaded = box_bake_high_poly_loaded != NULL && string_equals(box_bake_high_poly_loaded, box_bake_high_poly_path);
		if (!loaded && (box_bake_high_poly_tried == NULL || !string_equals(box_bake_high_poly_tried, box_bake_high_poly_path))) {
			box_bake_high_poly_tried = string_copy(box_bake_high_poly_path); // Do not retry a failing file every frame
			box_bake_load_high_poly();
			box_bake_preview_dirty = true;
		}
	}
	else if (!has_high && box_bake_high_poly_loaded != NULL) {
		box_bake_unload_high_poly();
	}

	if (box_bake_preview_dirty || box_bake_preview_image == NULL) {
		box_bake_preview_render();
	}

	i32 state = ui_image(box_bake_preview_image, 0xffffffff, -1);
	bool interacting = state == UI_STATE_DOWN || (g_ui->is_hovered && g_ui->input_wheel_delta != 0.0) || (g_ui->input_down && g_ui->is_hovered);
	if (box_bake_preview_fast && !g_ui->input_down) {
		// Interaction ended, refine with exact rays
		box_bake_preview_fast  = false;
		box_bake_preview_dirty = true;
		ui_box_hwnd->redraws   = 2;
	}
	if (interacting) {
		box_bake_preview_fast = true;
	}
	if (state == UI_STATE_DOWN && (g_ui->input_dx != 0.0 || g_ui->input_dy != 0.0)) {
		box_bake_preview_yaw -= g_ui->input_dx * 0.01;
		box_bake_preview_pitch = math_min(math_max(box_bake_preview_pitch + g_ui->input_dy * 0.01, -1.5), 1.5);
		box_bake_preview_dirty = true;
		ui_box_hwnd->redraws   = 2;
	}
	if (g_ui->is_hovered && g_ui->input_wheel_delta != 0.0) {
		box_bake_preview_zoom  = math_min(math_max(box_bake_preview_zoom * (1.0 + g_ui->input_wheel_delta * 0.1), 0.2), 5.0);
		box_bake_preview_dirty = true;
		ui_box_hwnd->redraws   = 2;
	}
	if (g_ui->is_hovered) {
		ui_tooltip(tr("Drag to orbit, wheel to zoom"));
	}

	ui_row4();
	ui_check(&box_bake_show_low, tr("Low Poly"), "");
	box_bake_preview_dirty |= ui_item_changed();
	g_ui->enabled = has_high;
	ui_check(&box_bake_show_high, tr("High Poly"), "");
	box_bake_preview_dirty |= ui_item_changed();
	g_ui->enabled = true;
	ui_check(&box_bake_show_cage, tr("Cage"), "");
	box_bake_preview_dirty |= ui_item_changed();
	g_ui->enabled = has_high;
	ui_check(&box_bake_show_miss, tr("Missed Rays"), "");
	box_bake_preview_dirty |= ui_item_changed();
	g_ui->enabled = true;

	f32 dist = box_bake_ray_dist;
	ui_slider(&box_bake_ray_dist, tr("Max Distance"), 0.001, 0.2, true, 1000.0, true, UI_ALIGN_RIGHT, true);
	if (g_ui->is_hovered) {
		ui_tooltip(tr("Cage size: ray search distance in front of and behind the low poly, relative to the mesh size"));
	}
	if (dist != box_bake_ray_dist) {
		box_bake_preview_dirty = true;
		box_bake_preview_fast  = g_ui->input_down; // Refined once the slider is released
	}

	if (!has_high) {
		ui_text(tr("Enable High Poly and pick a file to see the areas the rays miss"), UI_ALIGN_LEFT, 0);
	}
	else if (box_bake_show_miss) {
		any_map_t *vars = any_map_create();
		f32 ratio = bake_rays_miss_ratio() * 100.0;
		any_map_set(vars, "percent", string("%.1f", ratio));
		ui_text(vtr("Red: rays miss the high poly ({percent}% of rays), increase Max Distance", vars), UI_ALIGN_LEFT, 0);
		map_free(vars);
	}
}

static void box_bake_draw_buttons() {
	ui_separator(8, false);
	ui_row2();
	if (ui_icon_button(tr("Cancel"), ICON_CLOSE, UI_ALIGN_CENTER)) {
		ui_box_hide();
	}
	if (ui_icon_button(tr("Bake"), ICON_BAKE, UI_ALIGN_CENTER)) {
		if (string_equals(box_bake_folder, "")) {
			box_bake_pick_folder(&box_bake_folder_picked_and_run);
		}
		else {
			box_bake_run();
		}
	}
}

static void box_bake_draw_maps_column() {
	box_bake_draw_maps();
	box_bake_draw_high_poly();
	box_bake_draw_output();
}

static void box_bake_draw_settings_column() {
	box_bake_draw_device();
	box_bake_draw_common();
	box_bake_draw_settings();
}

// Settings on the left, 3D preview on the right, buttons below both
static void box_bake_draw_columns(void (*left)(void)) {
	f32 x0  = g_ui->_x;
	f32 y0  = g_ui->_y;
	f32 w   = g_ui->_w;
	f32 gap = 8 * UI_SCALE();
	f32 lw  = math_floor(w * 0.42);

	g_ui->_w = lw;
	left();
	f32 left_y = g_ui->_y;

	g_ui->_x = x0 + lw + gap;
	g_ui->_y = y0;
	g_ui->_w = w - lw - gap;
	box_bake_draw_preview();
	f32 right_y = g_ui->_y;

	g_ui->_x = x0;
	g_ui->_w = w;
	g_ui->_y = math_max(left_y, right_y);
	box_bake_draw_buttons();
}

static void box_bake_draw() {
	ui_box_click_to_hide = !box_bake_running;

	if (box_bake_running) {
		if (ui_tab(&box_bake_tab, tr("Bake Mesh Maps"), false, -1, false)) {
			box_bake_draw_progress();
		}
		return;
	}

	if (ui_tab(&box_bake_tab, tr("Maps"), false, -1, false)) {
		box_bake_draw_columns(&box_bake_draw_maps_column);
	}
	if (ui_tab(&box_bake_tab, tr("Settings"), false, -1, false)) {
		box_bake_draw_columns(&box_bake_draw_settings_column);
	}
}

// Release the high poly and the cpu bake data, they can take gigabytes
static void box_bake_on_hide() {
	if (box_bake_running) {
		return;
	}
	box_bake_unload_high_poly();
	bake_rays_end();
	box_bake_preview_low_ready = false;
}

void box_bake_show() {
	if (box_bake_prefix == NULL) {
		box_bake_prefix = string_copy(box_bake_default_prefix());
	}
	if (box_bake_high_poly_path == NULL) {
		box_bake_high_poly_path = "";
	}
	box_bake_preview_low_ready = false;
	if (box_bake_folder == NULL) {
		// Default to the project folder when saved
		char *path      = g_project->_->filepath;
		box_bake_folder = string_equals(path, "") ? "" : string_copy(substring(path, 0, string_last_index_of(path, PATH_SEP)));
	}
	ui_box_show_custom(&box_bake_draw, 1100, 720, &box_bake_on_hide, true, tr("Bake Mesh Maps"));
}
