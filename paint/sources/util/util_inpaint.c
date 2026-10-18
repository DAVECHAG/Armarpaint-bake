
#include "../global.h"

// Inpaint tool: a mask is painted on the mesh, on release the masked texels of the baked maps
// are rebuilt from the surrounding texels of the same uv island, then the png files are rewritten.
// Copy mode synthesizes texture from nearby patches, smooth mode diffuses the border inwards.

#define INPAINT_MODE_COPY   0
#define INPAINT_MODE_SMOOTH 1
#define INPAINT_PATCH       4 // Patch half size, 9x9 patches
#define INPAINT_PREVIEW     1024

i32 inpaint_mode   = INPAINT_MODE_COPY;
i32 inpaint_target = 0; // 0: every baked map of the mesh, n: the n-th of them

static u8   *inpaint_mask      = NULL;
static i32   inpaint_w         = 0;
static i32   inpaint_h         = 0;
static bool  inpaint_ready     = false;
static bool  inpaint_stroking  = false;
static bool  inpaint_has_mask  = false;
static bool  inpaint_has_last  = false;
static f32   inpaint_last[3];
static bool  inpaint_preview_dirty = false;
static bool  inpaint_was_active    = false;
static gpu_texture_t *inpaint_preview     = NULL;
static gpu_texture_t *inpaint_preview_old = NULL;

typedef struct inpaint_undo {
	asset_t *asset;
	u8      *pixels;
	i32      w;
	i32      h;
} inpaint_undo_t;

#define INPAINT_UNDO_MAX 16
static inpaint_undo_t inpaint_undo[INPAINT_UNDO_MAX];
static i32            inpaint_undo_count = 0;

// Baked maps of the active mesh: project textures named <mesh>_<map>.png
asset_t_array_t *inpaint_targets() {
	asset_t_array_t *list   = any_array_create_from_raw_tmp((void *[]){}, 0);
	asset_t_array_t *assets = g_project->_->assets;
	if (assets == NULL) {
		return list;
	}
	for (i32 k = -2; k < g_project->_->paint_objects->length; ++k) {
		char *name = k == -2   ? box_bake_prefix
		             : k == -1 ? (g_context->paint_object != NULL ? g_context->paint_object->base->name : NULL)
		                       : g_project->_->paint_objects->buffer[k]->base->name;
		if (name == NULL || string_equals(name, "")) {
			continue;
		}
		char *start = string_tmp("%s_", name);
		for (i32 i = 0; i < assets->length; ++i) {
			asset_t *a = assets->buffer[i];
			char *lower = to_lower_case(a->name);
			bool  png   = ends_with(lower, ".png");
			free(lower);
			if (a->image == NULL || !starts_with(a->name, start) || !png) {
				continue;
			}
			if (array_index_of(list, a) < 0) {
				any_array_push(list, a);
			}
		}
		if (list->length > 0) {
			break;
		}
	}
	return list;
}

static bool inpaint_is_vector_map(asset_t *a) {
	char *n      = to_lower_case(a->name);
	bool  vector = string_index_of(n, "normal") >= 0;
	free(n);
	return vector;
}

static u8 *inpaint_load_pixels(asset_t *a, i32 *w, i32 *h) {
	buffer_t *b = iron_load_blob(a->file);
	if (b == NULL) {
		return NULL;
	}
	int comp;
	u8 *p = stbi_load_from_memory(b->buffer, b->length, w, h, &comp, 4);
	free(b->buffer);
	free(b);
	return p;
}

// Prepares the uv layout at the resolution of the baked maps
static bool inpaint_prepare() {
	asset_t_array_t *targets = inpaint_targets();
	if (targets->length == 0 || box_bake_running) {
		return false;
	}
	asset_t *a = targets->buffer[0];
	i32      w = a->image->width;
	i32      h = a->image->height;
	if (inpaint_ready && inpaint_w == w && inpaint_h == h && bake_rays_inpaint_valid(w, h)) {
		return true;
	}
	if (!bake_rays_inpaint_begin(g_project->_->paint_objects, w, h)) {
		console_error(tr("Inpaint needs a low poly with a UV map"));
		return false;
	}
	free(inpaint_mask);
	inpaint_mask     = calloc(w * h, 1);
	inpaint_w        = w;
	inpaint_h        = h;
	inpaint_ready    = true;
	inpaint_has_mask = false;
	return true;
}

// Mouse ray in the bake space, the object matrix maps it to world space
static bool inpaint_hit(f32 *hit) {
	mesh_object_t *o = g_context->paint_object;
	if (o == NULL || o->data == NULL) {
		return false;
	}
	ray_t *ray = raycast_get_ray(mouse_view_x(), mouse_view_y(), scene_camera);
	mat4_t inv = mat4_inv(o->base->transform->world);
	vec4_t a   = vec4_apply_mat4((vec4_t){ray->origin.x, ray->origin.y, ray->origin.z, 1.0}, inv);
	vec4_t b   = vec4_apply_mat4((vec4_t){ray->origin.x + ray->dir.x, ray->origin.y + ray->dir.y, ray->origin.z + ray->dir.z, 1.0}, inv);
	f32    org[3] = {a.x, a.y, a.z};
	f32    dir[3] = {b.x - a.x, b.y - a.y, b.z - a.z};
	return bake_rays_pick(org, dir, hit);
}

// World brush radius, as the paint shader computes it, converted to the bake space
static f32 inpaint_radius() {
	mesh_object_t *o     = g_context->paint_object;
	f32            world = g_context->brush_radius * g_context->brush_nodes_radius / 15.0 * 2.0;
	mat4_t         m     = o->base->transform->world;
	f32            sx    = sqrtf(m.m[0] * m.m[0] + m.m[1] * m.m[1] + m.m[2] * m.m[2]); // World units per bake unit
	return sx > 0.0 ? world / sx : world;
}

static void inpaint_update_preview() {
	i32 pw = math_max(math_min(inpaint_w, INPAINT_PREVIEW), 1); // 1x1 until the first stroke sets the resolution
	i32 ph = math_max(math_min(inpaint_h, INPAINT_PREVIEW), 1);
	u8 *p  = calloc(pw * ph * 4, 1);
	if (inpaint_mask != NULL && inpaint_has_mask && inpaint_w > 0) {
		for (i32 y = 0; y < inpaint_h; ++y) {
			for (i32 x = 0; x < inpaint_w; ++x) {
				if (inpaint_mask[y * inpaint_w + x]) {
					i32 i = ((y * ph / inpaint_h) * pw + x * pw / inpaint_w) * 4;
					p[i] = p[i + 1] = p[i + 2] = 255;
				}
			}
		}
	}
	for (i32 i = 0; i < pw * ph; ++i) {
		p[i * 4 + 3] = 255;
	}
	buffer_t b = {.buffer = p, .length = pw * ph * 4};
	// The previous texture may still be bound this frame, release it one update later
	if (inpaint_preview_old != NULL) {
		gpu_delete_texture(inpaint_preview_old);
	}
	inpaint_preview_old = inpaint_preview;
	inpaint_preview     = gpu_create_texture_from_bytes(&b, pw, ph, GPU_TEXTURE_FORMAT_RGBA32);
	free(p);
	inpaint_preview_dirty = false;
	g_context->ddirty     = 2;
}

// Mask overlay for the viewport, NULL when the tool is not active
gpu_texture_t *inpaint_preview_image() {
	return g_context->tool == TOOL_TYPE_INPAINT ? inpaint_preview : NULL;
}

// Fill

typedef struct inpaint_job {
	u8  *px;    // RGBA8 pixels of the map being repaired
	u8  *known; // 1 for valid texels outside the mask, then for filled ones
	i32 *isl;   // Island per texel, -1 outside the uv layout
	i32 *src;   // Source texel copied into each texel, for coherent copy
	i32  w;
	i32  h;
} inpaint_job_t;

static bool inpaint_same(inpaint_job_t *j, i32 a, i32 b) {
	return j->isl[a] >= 0 && j->isl[a] == j->isl[b];
}

// Diffusion: onion peel from the border, then relaxation
static void inpaint_smooth(inpaint_job_t *j, i32 *todo, i32 count) {
	static const i32 dx[4] = {-1, 1, 0, 0};
	static const i32 dy[4] = {0, 0, -1, 1};
	i32              w = j->w, h = j->h;
	i32              left = count;
	u8              *next = malloc(count);
	for (i32 pass = 0; left > 0 && pass < 4096; ++pass) {
		memset(next, 0, count);
		i32 filled = 0;
		for (i32 n = 0; n < count; ++n) {
			i32 i = todo[n];
			if (j->known[i]) {
				continue;
			}
			i32 x = i % w, y = i / w;
			i32 sum[4] = {0, 0, 0, 0}, c = 0;
			for (i32 k = 0; k < 4; ++k) {
				i32 nx = x + dx[k], ny = y + dy[k];
				if (nx < 0 || ny < 0 || nx >= w || ny >= h) {
					continue;
				}
				i32 q = ny * w + nx;
				if (j->known[q] && inpaint_same(j, i, q)) {
					for (i32 ch = 0; ch < 4; ++ch) {
						sum[ch] += j->px[q * 4 + ch];
					}
					c++;
				}
			}
			if (c > 0) {
				for (i32 ch = 0; ch < 4; ++ch) {
					j->px[i * 4 + ch] = sum[ch] / c;
				}
				next[n] = 1;
				filled++;
			}
		}
		for (i32 n = 0; n < count; ++n) {
			if (next[n]) {
				j->known[todo[n]] = 1;
			}
		}
		left -= filled;
		if (filled == 0) {
			break;
		}
	}
	free(next);

	// Relaxation smooths the onion rings, Gauss-Seidel in place
	for (i32 it = 0; it < 200; ++it) {
		for (i32 n = 0; n < count; ++n) {
			i32 i = todo[n];
			i32 x = i % w, y = i / w;
			i32 sum[4] = {0, 0, 0, 0}, c = 0;
			for (i32 k = 0; k < 4; ++k) {
				i32 nx = x + dx[k], ny = y + dy[k];
				if (nx < 0 || ny < 0 || nx >= w || ny >= h) {
					continue;
				}
				i32 q = ny * w + nx;
				if (j->known[q] && inpaint_same(j, i, q)) {
					for (i32 ch = 0; ch < 4; ++ch) {
						sum[ch] += j->px[q * 4 + ch];
					}
					c++;
				}
			}
			if (c > 0) {
				for (i32 ch = 0; ch < 4; ++ch) {
					j->px[i * 4 + ch] = (sum[ch] + c / 2) / c;
				}
			}
		}
	}
}

// Sum of squared differences over the known texels of both patches, -1 when nothing to compare
static f32 inpaint_patch_cost(inpaint_job_t *j, i32 p, i32 s, f32 best) {
	i32 w = j->w, h = j->h;
	i32 px = p % w, py = p / w, sx = s % w, sy = s / w;
	f32 cost = 0.0;
	i32 n    = 0;
	for (i32 oy = -INPAINT_PATCH; oy <= INPAINT_PATCH; ++oy) {
		i32 ay = py + oy, by = sy + oy;
		if (ay < 0 || by < 0 || ay >= h || by >= h) {
			continue;
		}
		for (i32 ox = -INPAINT_PATCH; ox <= INPAINT_PATCH; ++ox) {
			i32 ax = px + ox, bx = sx + ox;
			if (ax < 0 || bx < 0 || ax >= w || bx >= w) {
				continue;
			}
			i32 a = ay * w + ax, b = by * w + bx;
			if (!j->known[a] || !j->known[b] || !inpaint_same(j, p, a) || !inpaint_same(j, s, b)) {
				continue;
			}
			for (i32 ch = 0; ch < 3; ++ch) {
				f32 d = (f32)j->px[a * 4 + ch] - (f32)j->px[b * 4 + ch];
				cost += d * d;
			}
			n++;
		}
		if (n > 8 && cost / n > best) {
			return 1e30; // Already worse than the best candidate
		}
	}
	return n > 0 ? cost / n : -1.0;
}

static u32 inpaint_rand_state = 1;
static u32 inpaint_rand() {
	inpaint_rand_state ^= inpaint_rand_state << 13;
	inpaint_rand_state ^= inpaint_rand_state >> 17;
	inpaint_rand_state ^= inpaint_rand_state << 5;
	return inpaint_rand_state;
}

// Texture synthesis: each texel of the fill front copies the center of the best matching
// patch, candidates are random texels around plus the continuations of the neighbours' sources
static void inpaint_copy(inpaint_job_t *j, i32 *todo, i32 count, i32 extent) {
	static const i32 dx[8] = {-1, 1, 0, 0, -1, 1, -1, 1};
	static const i32 dy[8] = {0, 0, -1, 1, -1, -1, 1, 1};
	i32              w = j->w, h = j->h;
	i32              search = math_max(48, extent * 3);
	u8              *front  = malloc(count);
	i32              left   = count;
	for (i32 pass = 0; left > 0 && pass < 4096; ++pass) {
		memset(front, 0, count);
		for (i32 n = 0; n < count; ++n) {
			i32 i = todo[n];
			if (j->known[i]) {
				continue;
			}
			i32 x = i % w, y = i / w;
			for (i32 k = 0; k < 8; ++k) {
				i32 nx = x + dx[k], ny = y + dy[k];
				if (nx >= 0 && ny >= 0 && nx < w && ny < h && j->known[ny * w + nx] && inpaint_same(j, i, ny * w + nx)) {
					front[n] = 1;
					break;
				}
			}
		}
		i32 filled = 0;
		for (i32 n = 0; n < count; ++n) {
			if (!front[n]) {
				continue;
			}
			i32 i = todo[n];
			i32 x = i % w, y = i / w;
			i32 best_s = -1;
			f32 best   = 1e30;
			// Coherence: continue the patch the neighbours were copied from
			for (i32 k = 0; k < 8; ++k) {
				i32 nx = x + dx[k], ny = y + dy[k];
				if (nx < 0 || ny < 0 || nx >= w || ny >= h) {
					continue;
				}
				i32 q = ny * w + nx;
				if (!j->known[q] || j->src[q] < 0) {
					continue;
				}
				i32 sx = j->src[q] % w - dx[k], sy = j->src[q] / w - dy[k];
				if (sx < 0 || sy < 0 || sx >= w || sy >= h) {
					continue;
				}
				i32 s = sy * w + sx;
				if (!j->known[s] || j->src[s] != s || !inpaint_same(j, i, s)) {
					continue;
				}
				f32 c = inpaint_patch_cost(j, i, s, best);
				if (c >= 0.0 && c < best) {
					best   = c;
					best_s = s;
				}
			}
			// Random candidates of the same island around the texel
			for (i32 r = 0, found = 0; r < 400 && found < 32; ++r) {
				i32 sx = x + (i32)(inpaint_rand() % (2 * search + 1)) - search;
				i32 sy = y + (i32)(inpaint_rand() % (2 * search + 1)) - search;
				if (sx < 0 || sy < 0 || sx >= w || sy >= h) {
					continue;
				}
				i32 s = sy * w + sx;
				if (!j->known[s] || j->src[s] != s || !inpaint_same(j, i, s)) {
					continue;
				}
				found++;
				f32 c = inpaint_patch_cost(j, i, s, best);
				if (c >= 0.0 && c < best) {
					best   = c;
					best_s = s;
				}
			}
			if (best_s < 0) {
				continue;
			}
			memcpy(j->px + i * 4, j->px + best_s * 4, 4);
			j->src[i]   = best_s;
			j->known[i] = 1;
			filled++;
		}
		left -= filled;
		if (filled == 0) {
			break;
		}
	}
	free(front);
	if (left > 0) {
		inpaint_smooth(j, todo, count); // Texels without any source, like a fully masked island
	}
}

// Vector maps store unit vectors, keep the filled ones unit length
static void inpaint_normalize(inpaint_job_t *j, i32 *todo, i32 count) {
	for (i32 n = 0; n < count; ++n) {
		u8 *p = j->px + todo[n] * 4;
		f32 v[3] = {p[0] / 127.5 - 1.0, p[1] / 127.5 - 1.0, p[2] / 127.5 - 1.0};
		f32 l    = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
		if (l < 1e-4) {
			continue;
		}
		for (i32 k = 0; k < 3; ++k) {
			p[k] = (u8)math_min(math_max((v[k] / l * 0.5 + 0.5) * 255.0 + 0.5, 0.0), 255.0);
		}
	}
}

static void inpaint_write(asset_t *a, u8 *px, i32 w, i32 h) {
	buffer_t b = {.buffer = px, .length = w * h * 4};
	iron_write_png(a->file, &b, w, h, 0);
	project_reimport_texture(a);
}

static void inpaint_apply() {
	asset_t_array_t *all = inpaint_targets();
	asset_t_array_t *targets = any_array_create_from_raw_tmp((void *[]){}, 0);
	if (inpaint_target > 0 && inpaint_target <= all->length) {
		any_array_push(targets, all->buffer[inpaint_target - 1]);
	}
	else {
		for (i32 i = 0; i < all->length; ++i) {
			any_array_push(targets, all->buffer[i]);
		}
	}

	i32 w = inpaint_w, h = inpaint_h;
	i32 count = 0;
	i32 minx = w, miny = h, maxx = 0, maxy = 0;
	for (i32 i = 0; i < w * h; ++i) {
		if (inpaint_mask[i] && bake_rays_texel_island(i) >= 0) {
			count++;
			minx = math_min(minx, i % w);
			maxx = math_max(maxx, i % w);
			miny = math_min(miny, i / w);
			maxy = math_max(maxy, i / w);
		}
	}
	if (count == 0) {
		return;
	}
	i32 *todo = malloc(sizeof(i32) * count);
	i32 *isl  = malloc(sizeof(i32) * w * h);
	for (i32 i = 0, n = 0; i < w * h; ++i) {
		isl[i] = bake_rays_texel_island(i);
		if (inpaint_mask[i] && isl[i] >= 0) {
			todo[n++] = i;
		}
	}
	i32 extent = math_max(maxx - minx, maxy - miny) + 1;

	// Free the previous undo states
	for (i32 i = 0; i < inpaint_undo_count; ++i) {
		free(inpaint_undo[i].pixels);
	}
	inpaint_undo_count = 0;

	i32 done = 0;
	for (i32 t = 0; t < targets->length; ++t) {
		asset_t *a = targets->buffer[t];
		i32      iw, ih;
		u8      *px = inpaint_load_pixels(a, &iw, &ih);
		if (px == NULL || iw != w || ih != h) {
			console_error(string("%s: %s", tr("Map size differs from the other baked maps, skipped"), a->name));
			if (px != NULL) {
				stbi_image_free(px);
			}
			continue;
		}
		if (inpaint_undo_count < INPAINT_UNDO_MAX) {
			u8 *copy = malloc(w * h * 4);
			memcpy(copy, px, w * h * 4);
			inpaint_undo[inpaint_undo_count++] = (inpaint_undo_t){.asset = a, .pixels = copy, .w = w, .h = h};
		}

		inpaint_job_t j = {.px = px, .isl = isl, .w = w, .h = h};
		j.known         = malloc(w * h);
		j.src           = malloc(sizeof(i32) * w * h);
		for (i32 i = 0; i < w * h; ++i) {
			j.known[i] = isl[i] >= 0 && !inpaint_mask[i];
			j.src[i]   = j.known[i] ? i : -1;
		}
		inpaint_rand_state = 0x9e3779b9u; // Same synthesis for every map, keeps the maps consistent
		if (inpaint_mode == INPAINT_MODE_SMOOTH) {
			inpaint_smooth(&j, todo, count);
		}
		else {
			inpaint_copy(&j, todo, count, extent);
		}
		if (inpaint_is_vector_map(a)) {
			inpaint_normalize(&j, todo, count);
		}
		inpaint_write(a, px, w, h);
		free(j.known);
		free(j.src);
		stbi_image_free(px);
		done++;
	}
	free(todo);
	free(isl);

	any_map_t *vars = any_map_create();
	any_map_set(vars, "count", i32_to_string(done));
	console_info(vtr("Inpaint: {count} maps repaired", vars));
	map_free(vars);
}

void inpaint_undo_last() {
	for (i32 i = 0; i < inpaint_undo_count; ++i) {
		inpaint_write(inpaint_undo[i].asset, inpaint_undo[i].pixels, inpaint_undo[i].w, inpaint_undo[i].h);
		free(inpaint_undo[i].pixels);
	}
	if (inpaint_undo_count > 0) {
		console_info(tr("Inpaint: last repair undone"));
	}
	inpaint_undo_count = 0;
}

bool inpaint_can_undo() {
	return inpaint_undo_count > 0;
}

static void inpaint_clear_mask() {
	if (inpaint_mask != NULL) {
		memset(inpaint_mask, 0, inpaint_w * inpaint_h);
	}
	inpaint_has_mask = false;
	inpaint_update_preview();
}

// Mouse over the 3D view, the toolbar and header excluded
static bool inpaint_point_in_view(f32 x, f32 y) {
	return x > ui_toolbar_w(false) && y > 0 && x < sys_w() && y < sys_h();
}

static bool inpaint_mouse_in_view() {
	return inpaint_point_in_view(mouse_view_x(), mouse_view_y());
}

static bool inpaint_press_failed = false;

// Called every frame
void inpaint_update(void *_) {
	bool active = g_context->tool == TOOL_TYPE_INPAINT;
	if (active != inpaint_was_active) {
		// Mask overlay is added to or removed from the viewport shader
		inpaint_was_active = active;
		inpaint_stroking   = false;
		if (active && inpaint_preview == NULL) {
			inpaint_update_preview();
		}
		make_material_parse_mesh_material();
		g_context->ddirty = 2;
	}
	if (!active) {
		return;
	}

	// A press started in the view begins the stroke, even when its first frame was missed
	if (!mouse_down("left")) {
		inpaint_press_failed = false;
	}
	bool press_in_view = inpaint_point_in_view(g_ui->input_started_x - sys_x(), g_ui->input_started_y - sys_y());
	if (!inpaint_stroking && !inpaint_press_failed && mouse_down("left") && press_in_view && inpaint_mouse_in_view() && !ui_menu_show && !g_context->brush_locked &&
	    !keyboard_down("alt")) {
		inpaint_press_failed = true; // Cleared on release, one attempt per press
		if (!inpaint_prepare()) {
			console_error(tr("Inpaint: no baked maps found for this mesh, bake them first"));
			return;
		}
		inpaint_stroking     = true;
		inpaint_press_failed = false;
		inpaint_has_last     = false;
	}
	if (inpaint_stroking && mouse_down("left")) {
		f32 hit[3];
		if (inpaint_hit(hit)) {
			f32 r = inpaint_radius();
			if (inpaint_has_last) {
				// Stamps along the segment so fast strokes stay continuous
				f32 d[3] = {hit[0] - inpaint_last[0], hit[1] - inpaint_last[1], hit[2] - inpaint_last[2]};
				f32 len  = sqrtf(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
				i32 steps = math_min((i32)(len / math_max(r * 0.4, 1e-6)), 64);
				for (i32 s = 1; s <= steps; ++s) {
					f32 t    = s / (f32)(steps + 1);
					f32 c[3] = {inpaint_last[0] + d[0] * t, inpaint_last[1] + d[1] * t, inpaint_last[2] + d[2] * t};
					bake_rays_stamp(inpaint_mask, c, r);
				}
			}
			bake_rays_stamp(inpaint_mask, hit, r);
			memcpy(inpaint_last, hit, sizeof(f32) * 3);
			inpaint_has_last      = true;
			inpaint_has_mask      = true;
			inpaint_preview_dirty = true;
		}
	}
	if (inpaint_preview_dirty) {
		inpaint_update_preview();
	}
	if (inpaint_stroking && !mouse_down("left")) {
		inpaint_stroking = false;
		if (inpaint_has_mask) {
			inpaint_apply();
		}
		inpaint_clear_mask();
	}
}

void inpaint_draw_header() {
	string_array_t *mode_combo = any_array_create_from_raw_tmp(
	    (void *[]){
	        tr("Copy"),
	        tr("Smooth"),
	    },
	    2);
	ui_combo(&inpaint_mode, mode_combo, tr("Mode"), true, UI_ALIGN_LEFT, true);
	if (g_ui->is_hovered) {
		ui_tooltip(tr("Copy: rebuilds the texture from nearby patches. Smooth: blends the surrounding values"));
	}

	asset_t_array_t *targets = inpaint_targets();
	string_array_t  *names   = any_array_create_from_raw_tmp((void *[]){}, 0);
	any_array_push(names, tr("All baked maps"));
	for (i32 i = 0; i < targets->length; ++i) {
		any_array_push(names, targets->buffer[i]->name);
	}
	if (inpaint_target > targets->length) {
		inpaint_target = 0;
	}
	ui_combo(&inpaint_target, names, tr("Maps"), false, UI_ALIGN_LEFT, true);
	if (g_ui->is_hovered) {
		ui_tooltip(tr("Baked maps repaired by the stroke"));
	}

	ui_slider(&g_context->brush_radius, tr("Radius"), 0.01, 2.0, true, 100.0, true, UI_ALIGN_RIGHT, true);

	g_ui->enabled = inpaint_can_undo();
	if (ui_icon_button(tr("Undo"), ICON_UNDO, UI_ALIGN_CENTER)) {
		inpaint_undo_last();
	}
	g_ui->enabled = true;
	if (targets->length == 0) {
		ui_text(tr("No baked maps for this mesh"), UI_ALIGN_LEFT, 0);
	}
}
