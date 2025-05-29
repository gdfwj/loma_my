import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import compiler
import ctypes
import numpy as np
import matplotlib.pyplot as plt

def np_to_vec3_memmove(arr: np.ndarray, vec3_type):
    arr = np.asarray(arr, dtype=np.float32, order='C')
    assert arr.shape == (3,)
    vec = vec3_type()
    ctypes.memmove(ctypes.addressof(vec), arr.ctypes.data, ctypes.sizeof(vec))
    return vec

if __name__ == '__main__':
    with open('loma_code/full_color.py') as f:
        structs, lib = compiler.compile(f.read(),
                                  target = 'c',
                                  output_filename = '_code/full_color')
    # print(dir(lib))

    # exit()

    w = 20
    h = 20
    img = np.zeros([h, w, 3], dtype = np.single)
    sphere_color = np.array([0.8, 0.3, 0.3])
    lib.raytrace(w, h, np_to_vec3_memmove(sphere_color, structs['Vec3']), img.ctypes.data_as(ctypes.POINTER(structs['Vec3'])))

    sphere_color_truth = np.array([0.3, 0.3, 0.8], dtype = np.single)
    img_truth = np.zeros([h, w, 3], dtype = np.single)
    lib.raytrace(w, h, np_to_vec3_memmove(sphere_color_truth, structs['Vec3']), img_truth.ctypes.data_as(ctypes.POINTER(structs['Vec3'])))
    plt.imshow(img_truth)
    plt.savefig(f'./truth.png')
    # plt.show()
    epochs = 200

    for epoch in range(epochs):
        img = np.zeros([h, w, 3], dtype = np.single)
        lib.raytrace(w, h, np_to_vec3_memmove(sphere_color, structs['Vec3']), img.ctypes.data_as(ctypes.POINTER(structs['Vec3'])))
        loss = np.mean(np.abs(img - img_truth))
        if epoch % 20 == 0:
            plt.imshow(img)
            plt.savefig(f'./loss_{epoch}.png')
        img = img - img_truth
        # print(img.mean())
        # exit()
        # dw, dh = ctypes.c_int(0), ctypes.c_int(0)
        # d_sphere_color = np.zeros([3], dtype = np.single)
        Vec3 = structs['Vec3']
        d_color = Vec3()
        d_color.x = d_color.y = d_color.z = 0.0

        lib.d_raytrace(w, h, np_to_vec3_memmove(sphere_color, structs['Vec3']), \
            ctypes.byref(d_color), \
            img.ctypes.data_as(ctypes.POINTER(structs['Vec3'])))

        d_sphere_color = np.array([d_color.x, d_color.y, d_color.z], dtype=np.float32)
        # print(d_sphere_color)

        sphere_color = sphere_color - 0.001 * d_sphere_color
        
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
        if loss < 1e-6:
            break