import sys, os, struct


def read_string(file):
    string_size = struct.unpack('<I', file.read(4))[0]
    return file.read(string_size).decode('ascii')

def read_strings(file, num):
    return tuple(read_string(file) for _ in range(num))

def read_float(file):
    return _read_single(file, "f")

def read_floats(file, num):
    return _read_multiple(file, "f", num)

def read_uint(file):
    return _read_single(file, "I")

def read_uints(file, num):
    return _read_multiple(file, "I", num)

def read_ushorts(file, num):
    return _read_multiple(file, "H", num)

def read_uchars(file, num):
    return _read_multiple(file, "B", num)

def _read_single(file, format):
    size = {
        "f": 4,
        "I": 4,
        "H": 2,
        "B": 1
    }[format]
    return struct.unpack("<" + format, file.read(size))[0]

def _read_multiple(file, format, num):
    return tuple(_read_single(file, format) for _ in range(num))


class Mesh:
    def __init__(self, submeshes):
        self._submeshes = submeshes

    def export_to_obj(self, obj_file, mtl_file, mtl_filepath):
        obj_file.write(f"# Exported from RW:ED\n")

        # Write reference to the material file (.mtl)
        obj_file.write(f"mtllib {os.path.basename(mtl_filepath)}\n")

        # Index counters for vertices, texture coords, and normals (OBJ indices are 1-based)
        vertex_offset = 1
        
        for submesh in self._submeshes:
            material, spec_map, positions, normals, colours, uvs, indices = submesh

            # Use the material from the material library
            obj_file.write(f"\nusemtl {material}\n")
            
            # Write a comment with the material name
            obj_file.write(f"\n# Material: {material}\n")
            
            # Write vertex positions
            for position in positions:
                obj_file.write(f"v {position[0]} {position[1]} {position[2]}\n")
            
            # Write texture coordinates (if present)
            for uv in uvs:
                obj_file.write(f"vt {uv[0]} {uv[1]}\n")
            
            # Write vertex normals (if present)
            for normal in normals:
                obj_file.write(f"vn {normal[0]} {normal[1]} {normal[2]}\n")

            # Write face indices (OBJ is 1-based index)
            for i in range(0, len(indices), 3):
                # The face format in OBJ: f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
                v1, v2, v3 = indices[i], indices[i+1], indices[i+2]
                obj_file.write(f"f {v1+vertex_offset}/{v1+vertex_offset}/{v1+vertex_offset} "
                        f"{v2+vertex_offset}/{v2+vertex_offset}/{v2+vertex_offset} "
                        f"{v3+vertex_offset}/{v3+vertex_offset}/{v3+vertex_offset}\n")

        print(f"Mesh successfully exported")

        # Write the .mtl file
        self.export_mtl(mtl_file)

    def export_mtl(self, f):
        f.write(f"# Exported from RW:ED\n")
        
        for submesh in self._submeshes:
            material, spec_map, positions, normals, colours, uvs, indices = submesh
            
            # Write material properties (for now, we assume simple properties)
            f.write(f"\nnewmtl {material}\n")
            f.write("Ka 1.000 1.000 1.000\n")  # Ambient color (white)
            f.write("Kd 1.000 1.000 1.000\n")  # Diffuse color (white)
            f.write("Ks 0.000 0.000 0.000\n")  # Specular color (black)
            f.write("d 1.0\n")  # Transparency (opaque)
            f.write("illum 2\n")  # Illumination model (default)
            f.write(f"map_Kd {material}.tga\n")  # Diffuse texture

            # Check if we have a specular map (assuming spec_map exists in your material)
            if spec_map:
                f.write(f"map_Ks {spec_map}.tga\n")  # Specular map
        
        print(f"Materials successfully exported")

    def export_to_rwm(f):
        f.write(b"MSH\x01")


    @classmethod
    def read_from_rwm_file(cls, f):
        assert f.read(4) == b"MSH\x01"
        num_mysteries = read_uint(f)
        print(f"Number of mysteries: {num_mysteries}")
        num_materials = read_uint(f)
        print(f"Number of materials: {num_materials}")
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
            print(f"\nMystery: {mystery}")
            mysteries.append(mystery)
            f.read(0x44)  # Who cares

        materials = []
        spec_maps = {}
        for _ in range(num_materials):
            material = read_string(f)
            print(f"\nMaterial: {material}")
            materials.append(material)
            spec_map = read_string(f)  # The specular map may be an empty string if not present
            if spec_map:
                print(f"Specular map: {spec_map}")
                spec_maps[material] = spec_map
            print(f"Bitfield maybe: {f.read(4)}")
            print(f"No idea: {read_float(f)}")
            print(f"Another number (flags?): {read_uint(f)}")

        submeshes = []
        for material in materials:  # Each material is a string
            positions = []
            normals = []
            colours = []
            uvs = []
            assert read_uint(f) == 0
            num_verts = read_uint(f)
            print(f"Number of vertices for this material ({material}): {num_verts}")

            for _ in range(num_verts):
                positions.append(read_floats(f, 3))
                normals.append(read_floats(f, 3))
                colours.append((c / 255. for c in read_uchars(f, 4)))
                uv = read_floats(f, 2)
                uvs.append((uv[0], -uv[1]))  # Flip V

            assert read_uints(f, 2) == (1,0)
            num_tris = read_uint(f)
            print(f"Number of triangles: {num_tris}")
            indices = read_ushorts(f, num_tris*3)
            submeshes.append((material, spec_maps.get(material), positions, normals, colours, uvs, indices))

        print(f"\nFinished at {hex(f.tell())}")
        next_part = f.read(4)
        print("Dynamics part", "follows" if next_part else "does NOT follow")
        assert next_part == b"DYS\x01" or not next_part

        return Mesh(submeshes)
    
    @classmethod
    def read_from_obj_file(cls, f):
        submeshes = []
        current_mat = None
        current_spec_map = None
        current_positions = []
        current_normals = []
        current_colours = []
        current_uvs = []
        current_indices = []

        def add_current_submesh():
            assert len(current_positions) == len(current_normals) == len(current_uvs)
            current_colours = [(1., 1., 1., 1.) for _ in current_positions]
            submeshes.append((current_mat,
                              current_spec_map,
                              current_positions,
                              current_normals,
                              current_colours,
                              current_uvs,
                              current_indices))
            submeshes = []
            current_mat = None
            current_spec_map = None
            current_positions = []
            current_normals = []
            current_colours = []
            current_uvs = []
            current_indices = []

        for line in f.readlines():
            words = line.split()
            if not words or words[0].startswith("#"):
                continue

            elif words[0] == "mtllib":
                continue  # This is only useful for the spec map, so let's not bother for now

            elif words[0] == "usemtl":
                if current_mat:
                    add_current_submesh()
                current_mat = words[1]

            elif words[0] == "f":
                assert len(words) == 4 and all(v.split("/")[0] == v.split("/")[1] == v.split("/")[2] for v in words[1:])
                for v in words[1:]:
                    current_indices.append(int(v.split("/")[0]))

            else:
                thing = tuple(float(word) for word in words[1:])
                {
                    "v": current_positions,
                    "vn": current_normals,
                    "vt": current_uvs
                }[words[0]].append(thing)

        add_current_submesh()
        return Mesh(submeshes)


if __name__ == "__main__":
    in_path = sys.argv[1]
    print(f"Reading {in_path}".center(80, "="))
    basename, extension = os.path.splitext(in_path)
    with open(in_path, "rb" if extension == ".rwm" else "r") as obj_file:
        if extension == ".rwm":
            m = Mesh.read_from_rwm_file(obj_file)
            with open(basename + ".obj", 'w') as f1:
                with open(basename + ".mtl", 'w') as f2:
                    m.export_to_obj(f1, f2, basename + ".mtl")
        else:
            m = Mesh.read_from_obj_file(obj_file)
            with open(basename + ".rwm", 'wb') as obj_file:
                m.export_to_rwm(obj_file)