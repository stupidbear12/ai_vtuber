import struct, zlib, os

W, H = 32, 32
pixels = []
cx, cy = W / 2.0, H / 2.0
r = 14.0
for y in range(H):
    for x in range(W):
        dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
        if dist <= r:
            pixels += [0xe8, 0x6b, 0xce, 0xff]
        else:
            pixels += [0, 0, 0, 0]

raw_data = b""
for y in range(H):
    raw_data += b"\x00" + bytes(pixels[y * W * 4:(y + 1) * W * 4])


def make_chunk(name, data):
    c = zlib.crc32(name + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + name + data + struct.pack(">I", c)


sig = b"\x89PNG\r\n\x1a\n"
ihdr_data = struct.pack(">II", W, H) + bytes([8, 6, 0, 0, 0])
ihdr = make_chunk(b"IHDR", ihdr_data)
idat = make_chunk(b"IDAT", zlib.compress(raw_data))
iend = make_chunk(b"IEND", b"")

png = sig + ihdr + idat + iend
out = os.path.join(os.path.dirname(__file__), "icon.png")
with open(out, "wb") as f:
    f.write(png)
print(f"icon.png created: {len(png)} bytes at {out}")
