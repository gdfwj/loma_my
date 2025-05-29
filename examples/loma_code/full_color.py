class Vec3:
    x: float
    y: float
    z: float

class Sphere:
    center: Vec3
    radius: float
    color: Vec3  # 每个球的颜色

class Ray:
    org: Vec3
    dir: Vec3

def make_vec3(x: In[float], y: In[float], z: In[float]) -> Vec3:
    ret: Vec3
    ret.x = x
    ret.y = y
    ret.z = z
    return ret

def add(a: In[Vec3], b: In[Vec3]) -> Vec3:
    return make_vec3(a.x + b.x, a.y + b.y, a.z + b.z)

def sub(a: In[Vec3], b: In[Vec3]) -> Vec3:
    return make_vec3(a.x - b.x, a.y - b.y, a.z - b.z)

def mul(a: In[float], b: In[Vec3]) -> Vec3:
    return make_vec3(a * b.x, a * b.y, a * b.z)

def dot(a: In[Vec3], b: In[Vec3]) -> float:
    return a.x * b.x + a.y * b.y + a.z * b.z

def normalize(v: In[Vec3]) -> Vec3:
    l: float = sqrt(dot(v, v))
    return make_vec3(v.x / l, v.y / l, v.z / l)

def sphere_isect(sph: In[Sphere], ray: In[Ray]) -> float:
    oc: Vec3 = sub(ray.org, sph.center)
    a: float = dot(ray.dir, ray.dir)
    b: float = 2 * dot(oc, ray.dir)
    c: float = dot(oc, oc) - sph.radius * sph.radius
    discriminant: float = b * b - 4 * a * c
    ret_dist: float = 0
    if discriminant < 0:
        ret_dist = -1
    else:
        ret_dist = (-b - sqrt(discriminant)) / (2 * a)
    return ret_dist

def ray_color(ray: In[Ray], sph: In[Sphere]) -> Vec3:
    t: float = sphere_isect(sph, ray)

    white: Vec3 = make_vec3(1, 1, 1)
    blue: Vec3 = make_vec3(0.5, 0.7, 1)
    a: float
    ret_color: Vec3

    if t > 0:
        ret_color = sph.color
    else:
        a = 0.5 * ray.dir.y + 1
        ret_color = add(mul((1 - a), white), mul(a, blue))

    return ret_color

def raytrace(w: In[int], h: In[int], sphere_color: In[Vec3], image: Out[Array[Vec3]]):
    # 定义球体
    sph: Sphere
    sph.center = make_vec3(0, 0, -1)
    sph.radius = 0.5
    sph.color = sphere_color  # 颜色作为参数传入

    # 相机与视口配置
    aspect_ratio: float = int2float(w) / int2float(h)
    focal_length: float = 1.0
    viewport_height: float = 2.0
    viewport_width: float = viewport_height * aspect_ratio
    camera_center: Vec3 = make_vec3(0, 0, 0)
    pixel_delta_u: Vec3 = make_vec3(viewport_width / w, 0, 0)
    pixel_delta_v: Vec3 = make_vec3(0, -viewport_height / h, 0)
    viewport_upper_left: Vec3 = make_vec3(
        camera_center.x - viewport_width / 2,
        camera_center.y + viewport_height / 2,
        camera_center.z - focal_length
    )
    pixel00_loc: Vec3 = viewport_upper_left
    pixel00_loc.x = pixel00_loc.x + pixel_delta_u.x / 2
    pixel00_loc.y = pixel00_loc.y - pixel_delta_v.y / 2

    y: int = 0
    x: int
    pixel_center: Vec3
    ray: Ray
    while (y < h, max_iter := 20):
        x = 0
        while (x < w, max_iter := 20):
            pixel_center = add(add(pixel00_loc, mul(x, pixel_delta_u)), mul(y, pixel_delta_v))
            ray.org = camera_center
            ray.dir = normalize(sub(pixel_center, camera_center))
            image[w * y + x] = ray_color(ray, sph)
            x = x + 1
        y = y + 1

d_raytrace = rev_diff(raytrace)