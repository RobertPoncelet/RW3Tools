import sys, os, struct
from functools import reduce
from pxr import Usd, UsdGeom, Gf, Sdf


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


def write_string(file, string):
    write_uint(file, len(string))
    file.write(string.encode("ascii"))

def write_uint(file, value):
    _write_single(file, "I", value)

def write_uints(file, values):
    _write_multi(file, "I", values)

def write_float(file, value):
    _write_single(file, "f", value)

def write_floats(file, values):
    _write_multi(file, "f", values)

def write_ushorts(file, values):
    _write_multi(file, "H", values)

def write_uchars(file, values):
    _write_multi(file, "B", values)

def _write_single(file, format, value):
    file.write(struct.pack("<" + format, value))

def _write_multi(file, format, values):
    file.write(struct.pack("<" + (format * len(values)), values))


class Mesh:
    def __init__(self, submeshes):
        self._submeshes = submeshes

    def write_to_rwm(self, f):
        f.write(b"MSH\x01")
        write_uint(f, 0)  # num_mysteries
        write_uint(f, len(self._submeshes))
        all_positions = reduce(lambda a, b: a.extend(b), [submesh[2] for submesh in self._submeshes])
        all_indices = reduce(lambda a, b: a.extend(b), [submesh[6] for submesh in self._submeshes])
        write_uint(f, len(all_positions))
        write_uint(f, len(all_indices) / 3)
        write_uint(f, 1)  # ???
        minx = min(v[0] for v in all_positions)
        miny = min(v[1] for v in all_positions)
        minz = min(v[2] for v in all_positions)
        maxx = max(v[0] for v in all_positions)
        maxy = max(v[1] for v in all_positions)
        maxz = max(v[2] for v in all_positions)
        write_floats(f, (minx, miny, minz))
        write_floats(f, (maxx, maxy, maxz))
        write_floats(f, ((minx+maxx)/2, (miny+maxy)/2, (minz+maxz)/2))
        write_float(f, 30.)  # Mass?

        # TODO: write mysteries here

        for submesh in self._submeshes:
            material = submesh[0]
            spec_map = submesh[1]
            write_string(f, material)
            write_string(f, spec_map or "")
            f.write(b"\xcc\xcc\xcc\xff")
            write_float(f, 30.)  # ???
            write_uint(f, 0)  # Also dunno, flags?
        
        for submesh in self._submeshes:
            write_uint(f, 0)
            num_verts = len(submesh[2])  # Positions
            write_uint(f, num_verts)
            for i in range(num_verts):
                write_floats(f, submesh[2][i])
                write_floats(f, submesh[3][i])
                write_uchars(f, (int(c*255) for c in submesh[4][i]))
                uv = submesh[5][i]
                write_floats(f, (uv[0], -uv[1]))  # Flip V
            write_uints(f, (1, 0))
            write_uint(f, len(submesh[6])/3)
            write_ushorts(f, submesh[6])

    @classmethod
    def read_from_rwm_file(cls, f):
        assert f.read(4) == b"MSH\x01"
        num_mysteries = read_uint(f)
        print(f"Number of mysteries: {num_mysteries}")
        num_materials = read_uint(f)
        print(f"Number of materials: {num_materials}")
        print(f"Total number of verts: {read_uint(f)}")
        print(f"Total number of triangles: {read_uint(f)}")
        print(f"A low number, but NOT the number of materials: {read_uint(f)}")
        print(f"Bounding box min: {read_floats(f, 3)}")
        print(f"Bounding box max: {read_floats(f, 3)}")
        print(f"Another point (origin or centre of mass?): {read_floats(f, 3)}")
        print(f"Something else (mass?): {read_float(f)}")

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
        num_indices = 0
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

            assert read_uints(f, 2) == (1, 0)
            num_tris = read_uint(f)
            print(f"Number of triangles: {num_tris}")
            indices = [i - num_indices for i in read_ushorts(f, num_tris*3)]
            num_indices += max(indices) + 1
            submeshes.append((material, spec_maps.get(material), positions, normals, colours, uvs, indices))

        print(f"\nFinished at {hex(f.tell())}")
        next_part = f.read(4)
        print("Dynamics part", "follows" if next_part else "does NOT follow")
        assert next_part == b"DYS\x01" or not next_part

        return Mesh(submeshes)
    
    def export_to_usd(self, filepath):
        # Create a new USD stage
        stage = Usd.Stage.CreateNew(filepath)
        
        # Create the root Xform (transform) node for the mesh
        root_prim = UsdGeom.Xform.Define(stage, "/Root")
        
        # Loop through submeshes and export them as individual meshes
        for submesh_idx, submesh in enumerate(self._submeshes):
            material, spec_map, positions, normals, colours, uvs, indices = submesh
            
            # Create a mesh node under /Root for each submesh
            mesh_path = f"/Root/Submesh{submesh_idx}"
            mesh_prim = UsdGeom.Mesh.Define(stage, mesh_path)
            
            # Set vertex positions
            points = [Gf.Vec3f(p[0], p[1], p[2]) for p in positions]
            mesh_prim.GetPointsAttr().Set(points)
            
            # Set normals
            mesh_prim.GetNormalsAttr().Set([Gf.Vec3f(n[0], n[1], n[2]) for n in normals])
            
            # # Set UVs
            tex_coords_attr = UsdGeom.PrimvarsAPI(mesh_prim).CreatePrimvar("st",
                                                                           Sdf.ValueTypeNames.TexCoord2fArray,
                                                                           interpolation="vertex")
            tex_coords_attr.Set([Gf.Vec2f(uv[0], uv[1]) for uv in uvs])

            # Create face vertex counts (all triangles, so each face has 3 vertices)
            mesh_prim.CreateFaceVertexCountsAttr([3] * (len(indices) // 3))
    
            # Set face vertex indices
            mesh_prim.CreateFaceVertexIndicesAttr(indices)
            
        # Save the stage
        stage.GetRootLayer().Save()

        print(f"Mesh successfully exported to USD: {filepath}")

    @classmethod
    def import_from_usd(self, filepath):
        # Load the USD stage
        stage = Usd.Stage.Open(filepath)
        root_prim = stage.GetPrimAtPath("/Root")
        
        submeshes = []

        # Iterate over submesh nodes
        for child in root_prim.GetChildren():
            if not child.IsA(UsdGeom.Mesh):
                continue  # Skip if it's not a mesh

            mesh_prim = UsdGeom.Mesh(child)
            
            # Read vertex positions
            positions_attr = mesh_prim.GetPointsAttr()
            positions = positions_attr.Get()

            # Read normals
            normals_attr = mesh_prim.GetNormalsAttr()
            normals = normals_attr.Get() if normals_attr else []

            # Read UVs
            tex_coords_primvar = mesh_prim.GetPrimvar("st")
            uvs = tex_coords_primvar.Get() if tex_coords_primvar else []

            # Read face indices
            indices_attr = mesh_prim.GetFaceVertexIndicesAttr()
            indices = indices_attr.Get()

            # For simplicity, let's assume no per-face materials for now
            material = "DefaultMaterial"

            submeshes.append((material, positions, normals, [], uvs, indices))

        print(f"Mesh successfully imported from USD: {filepath}")
        return Mesh(submeshes)


if __name__ == "__main__":
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    print(f"Converting {in_path} => {out_path}".center(80, "="))
    basename, in_extension = os.path.splitext(in_path)
    if in_extension == ".rwm":
        with open(in_path, "rb") as in_file:
            m = Mesh.read_from_rwm_file(in_file)
            m.export_to_usd(out_path)
    else:
        m = Mesh.import_from_usd(in_path)
        with open(out_path, "wb") as out_file:
            m.write_to_rwm(out_file)