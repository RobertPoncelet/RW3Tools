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


def read_uchars(file, num):
    return struct.unpack_from('<' + 'B' * num, file.read(num))


class Mesh:
    def __init__(self, submeshes):
        self._submeshes = submeshes

    def export_to_obj(self, filepath):
        with open(filepath, 'w') as f:
            f.write(f"# Exported OBJ file from Mesh class\n")

            # Write reference to the material file (.mtl)
            mtl_filepath = filepath.replace(".obj", ".mtl")
            f.write(f"mtllib {os.path.basename(mtl_filepath)}\n")

            # Index counters for vertices, texture coords, and normals (OBJ indices are 1-based)
            vertex_offset = 1
            
            for submesh in self._submeshes:
                material, spec_map, positions, normals, colours, uvs, indices = submesh

                # Use the material from the material library
                f.write(f"\nusemtl {material}\n")
                
                # Write a comment with the material name
                f.write(f"\n# Material: {material}\n")
                
                # Write vertex positions
                for position in positions:
                    f.write(f"v {position[0]} {position[1]} {position[2]}\n")
                
                # Write texture coordinates (if present)
                for uv in uvs:
                    f.write(f"vt {uv[0]} {uv[1]}\n")
                
                # Write vertex normals (if present)
                for normal in normals:
                    f.write(f"vn {normal[0]} {normal[1]} {normal[2]}\n")

                # Write face indices (OBJ is 1-based index)
                for i in range(0, len(indices), 3):
                    # The face format in OBJ: f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
                    v1, v2, v3 = indices[i], indices[i+1], indices[i+2]
                    f.write(f"f {v1+vertex_offset}/{v1+vertex_offset}/{v1+vertex_offset} "
                            f"{v2+vertex_offset}/{v2+vertex_offset}/{v2+vertex_offset} "
                            f"{v3+vertex_offset}/{v3+vertex_offset}/{v3+vertex_offset}\n")

        print(f"Mesh successfully exported to {filepath}")

        # Write the .mtl file
        self.export_mtl(mtl_filepath)

    def export_mtl(self, filepath):
        with open(filepath, 'w') as f:
            f.write(f"# Exported MTL file from Mesh class\n")
            
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
        
        print(f"Materials successfully exported to {filepath}")

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
        for _ in range(num_materials):
            material = read_string(f)
            print(f"\nMaterial: {material}")
            materials.append(material)
            spec_map = read_string(f)  # The specular map may be an empty string if not present
            if spec_map:
                print(f"Specular map: {spec_map}")
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
            submeshes.append((material, spec_map, positions, normals, colours, uvs, indices))

        print(f"\nFinished at {hex(f.tell())}")
        next_part = f.read(4)
        print("Dynamics part", "follows" if next_part else "does NOT follow")
        assert next_part == b"DYS\x01" or not next_part

        return Mesh(submeshes)

if __name__ == "__main__":
    in_path = sys.argv[1]
    print(f"Reading {in_path}".center(80, "="))
    with open(in_path, "rb") as f:
        m = Mesh.read_from_rwm_file(f)
    out_path = os.path.splitext(in_path)[0] + ".obj"
    m.export_to_obj(out_path)