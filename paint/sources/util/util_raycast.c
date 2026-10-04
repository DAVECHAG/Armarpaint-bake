
#include "../global.h"

vec4_t raycast_aabb_mouse(object_t *object) {
	return raycast_box_intersect(object->transform, mouse_x, mouse_y, scene_camera);
}

bool point_in_aabb(object_t *object, vec4_t point) {
	transform_t *t  = object->transform;
	f32          cx = transform_world_x(t);
	f32          cy = transform_world_y(t);
	f32          cz = transform_world_z(t);
	f32          hx = t->dim.x / 2;
	f32          hy = t->dim.y / 2;
	f32          hz = t->dim.z / 2;
	return point.x >= cx - hx && point.x <= cx + hx && point.y >= cy - hy && point.y <= cy + hy && point.z >= cz - hz && point.z <= cz + hz;
}
