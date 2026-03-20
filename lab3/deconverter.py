import sys
import struct
from PIL import Image
import ctypes

pal = [
    (0, 0, 0), (128, 128, 128), (192, 192, 192), (255, 255, 255),
    (255, 0, 255), (128, 0, 128), (255, 0, 0), (128, 0, 0),
    (205, 92, 92), (240, 128, 128), (250, 128, 114), (233, 150, 122),
    (205, 92, 92), (240, 128, 128), (250, 128, 114), (233, 150, 122),
    (173, 255, 47), (127, 255, 0), (124, 252, 0), (0, 255, 0),
    (50, 205, 50), (152, 251, 152), (144, 238, 144), (0, 250, 154),
    (0, 255, 127), (60, 179, 113), (46, 139, 87), (34, 139, 34),
    (0, 128, 0), (0, 100, 0), (154, 205, 50), (107, 142, 35),
    (128, 128, 0), (85, 107, 47), (102, 205, 170), (143, 188, 143),
    (32, 178, 170), (0, 139, 139), (0, 128, 128)
]
def to_img_alfa(src, dst):
    with open(src, 'rb') as fin:
        (w, h) = struct.unpack('ii', fin.read(8))
        buff = ctypes.create_string_buffer(4 * w * h)
        fin.readinto(buff)

    img = Image.new('RGB', (w, h))
    pix = img.load()
    offset = 0
    sp = len(pal)
    for j in range(h):
        for i in range(w):
            (_, _, _, a) = struct.unpack_from('4B', buff, offset)
            pix[i, j] = pal[a % sp]
            offset += 4

    img.save(dst)


input_filename = "/home/arnemkova/PGP_PPT/lab3/out.data"
output_filename = input("output filename: ")
to_img_alfa(input_filename, output_filename)
