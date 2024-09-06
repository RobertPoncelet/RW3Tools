import sys, os, struct


def read_string(file):
    string_size = struct.unpack_from('<I', file.read(4))[0]
    return file.read(string_size).decode('ascii')


def read_strings(file, num):
    return tuple(read_string(file) for _ in range(num))


def read_float(file):
    return struct.unpack_from('<f', file.read(4))[0]


def read_floats(file, num):
    return struct.unpack_from('<' + 'f' * num, file.read(4 * num))


def read_uint(file):
    return struct.unpack_from('<I', file.read(4))[0]


def read_uints(file, num):
    return struct.unpack_from('<' + 'I' * num, file.read(4 * num))


def read_ushorts(file, num):
    return struct.unpack_from('<' + 'H' * num, file.read(2 * num))


class Mesh:
    def __init__():
        ...

    @classmethod
    def read_from_rwm_file(cls, f):
        assert f.read(4) == b"MSH\x01"
        num_mysteries = read_uint(f)
        print(f"Number of mysteries: {num_mysteries}")
        num_mats = read_uint(f)
        print(f"Number of materials: {num_mats}")
        print(f"Total number of verts: {read_uint(f)}")
        print(f"Another number: {read_uint(f)}")
        print(f"A low number, but NOT the number of materials: {read_uint(f)}")
        print(f"A point:                 {read_floats(f, 3)}")
        print(f"Another point:           {read_floats(f, 3)}")
        print(f"Another point (origin?): {read_floats(f, 3)}")
        print(f"Something else: {read_float(f)}")

        mysteries = []
        for _ in range(num_mysteries):
            mystery = read_string(f)
            print(f"\nmystery: {mystery}")
            mysteries.append(mystery)
            f.read(0x44)  # Who cares

        mats = []
        for _ in range(num_mats):
            mat = read_string(f)
            print(f"\nMaterial: {mat}")
            mats.append(mat)
            spec_map = read_string(f)
            if spec_map:
                print(f"Specular map: {spec_map}")
            print(f"Bitfield maybe: {f.read(4)}")
            print(f"No idea: {read_float(f)}")
            print(f"Another number (flags?): {read_uint(f)}")

        for mat in mats:
            print(f"\nProbably zero: {read_uint(f)}")
            num_verts = read_uint(f)
            print(f"Number of vertices for this material ({mat}): {num_verts}")

            for _ in range(num_verts):
                print(f"\nVertex position: {read_floats(f, 3)}")
                print(f"Vertex normal: {read_floats(f, 3)}")
                print(f"Vertex colour(?): {f.read(4)}")
                #assert f.read(4) == b"\xff" * 4
                print(f"UV: {read_floats(f, 2)}")

            print(f"\nDunno: {read_uints(f, 2)}")
            num_tris = read_uint(f)
            print(f"Number of triangles: {num_tris}")
            print(f"Indices: {read_ushorts(f, num_tris*3)}")

        print(f"\nFinished at {hex(f.tell())}")
        next_part = f.read(4)
        print("Dynamics part", "follows" if next_part else "does NOT follow")
        assert next_part == b"DYS\x01" or not next_part

if __name__ == "__main__":
    print(f"Reading {sys.argv[1]}".center(80, "="))
    with open(sys.argv[1], "rb") as f:
        m = Mesh.read_from_rwm_file(f)