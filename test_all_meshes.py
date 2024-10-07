from rw3tools.mesh import Mesh
from glob import glob
import os, sys
from multiprocessing import Pool, cpu_count

def test_file(rwm_path):
    print("Reading", rwm_path)
    usd_path = rwm_path[:-4] + ".tmp.usd"
    rwm_path2 = rwm_path[:-4] + ".tmp.rwm"
    try:
        with open(rwm_path, "rb") as f:
            m = Mesh.read_from_rwm(f)
        mesh_type = m._mesh_type
        m.export_to_usd(usd_path)
        del m

        m2 = Mesh.import_from_usd(usd_path, mesh_type)
        with open(rwm_path2, "wb") as f:
            m2.write_to_rwm(f)
        del m2

        # Validate the one we just wrote
        with open(rwm_path2, "rb") as f:
            m3 = Mesh.read_from_rwm(f)
        del m3
    except Mesh.UnsupportedMeshError:
        print("Unsupported, skipping")
    finally:
        if os.path.isfile(rwm_path2):
            os.remove(rwm_path2)
        if os.path.isfile(usd_path):
            os.remove(usd_path)

if __name__ == "__main__":
    paths = glob(sys.argv[1])
    #paths.remove("extracted/Graphics\\staticsphere.rwm")  # Weird material format?
    with Pool(cpu_count()) as pool:
        pool.map(test_file, paths)