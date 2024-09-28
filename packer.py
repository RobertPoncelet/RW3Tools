import argparse
import os
import re
import struct
import subprocess
from multiprocessing import Pool, cpu_count
from shutil import which


def byte_pad_string(string):
    string_bytes = string.encode('ascii') + b'\x00'
    padding_length = (4 - (len(string_bytes) % 4)) % 4
    return string_bytes + b'\x00' * padding_length


def to_compressed_path(path):
    return os.path.join(os.path.dirname(path), ".compressed", f"{os.path.basename(path)}.rnc")


def compressed(path, ignore_cache=False):
    compressed_path = to_compressed_path(path)
    if (os.path.isfile(compressed_path)
        and not ignore_cache and os.path.getmtime(path) < os.path.getmtime(compressed_path)):
            # All conditions are satisfied for us to use the cached version
            return compressed_path
    
    executable = "rnc_lib"
    if not which(executable):
        raise RuntimeError(f"{executable}.exe is missing from both the current directory and your "
                           "PATH. Ensure RNC ProPackED is available in one of these locations, or "
                           "use the --no-compress flag.")
    
    os.makedirs(os.path.dirname(compressed_path), exist_ok=True)

    print(f"Compressing {path}...")
    command = [
        executable, 'p', path, compressed_path, '-m=1'
    ]
    subprocess.check_call(command, stdout=subprocess.DEVNULL)
    return compressed_path


class IndexFileEntry:
    exclude_list = None

    def __init__(self, filename, offset, unpacked_size, packed_size, is_compressed, is_ghost_file):
        self.filename = filename
        self._offset = offset
        self._unpacked_size = unpacked_size
        self._packed_size = packed_size
        self._is_compressed = is_compressed
        self._is_ghost_file = is_ghost_file
    
    def entry_size(self):
        return 16 + len(byte_pad_string(self.filename))
    
    def offset(self):
        if self._offset == None:
            raise ValueError("f{self.filename} has no offset!")
        return self._offset
    
    def update_from_data_file(self, data_file):
        # For some reason, certain "ghost" files are marked as size 0 in the index, even though
        # they exist in the data file. In these cases, we need to check the data file itself.
        if not self.is_ghost_file():
            return

        data_file.seek(self.offset())
        header = data_file.read(4)
        if header == b"RNC\x01":  # RNC compression
            self._is_compressed = True
            self._unpacked_size = struct.unpack_from('>I', data_file.read(4))[0]
            print(f"Real uncompressed size of {self.filename}: {self._unpacked_size}")
            self._packed_size = struct.unpack_from('>I', data_file.read(4))[0]
            self._packed_size += 18  # Account for the RNC header
            print(f"Real compressed size of {self.filename}: {self._packed_size}")
        else:
            assert not self.is_compressed()
            # Fuck knows
            self._packed_size = self._unpacked_size = 0x800

    def write_packed_asset_to_data_file(self, packed_data, data_file, current_offset):
        # This is mainly so we can update our offset
        self._offset = current_offset
        assert len(packed_data) == self.packed_data_size()
        data_file.write(packed_data)
        CHUNK_SIZE = 4  # Use 0x80 to get the original 2KB optimised for CD reading
        padding_length = (CHUNK_SIZE - (len(packed_data) % CHUNK_SIZE)) % CHUNK_SIZE
        data_file.write(b'\x00' * padding_length)
        current_offset = self.offset() + self.packed_data_size() + padding_length
        return current_offset

    def is_ghost_file(self):
        return self._is_ghost_file
    
    def _file_size(self, compressed):
        size = self._packed_size if compressed else self._unpacked_size
        if not size:
            raise ValueError("Size can't be 0! Should probably update this entry from the data file first.")
        return size

    def unpacked_data_size(self):
        return self._file_size(False)
    
    def packed_data_size(self):
        return self._file_size(True)

    def is_compressed(self):
        return self._is_compressed

    def write_to_file(self, file):
        # Serialize the file entry and filename
        if self.is_ghost_file():
            unpacked_size = packed_size = 0
        else:
            unpacked_size = self.unpacked_data_size()
            packed_size = self.packed_data_size()

        if self.is_compressed():
            packed_compression = packed_size | (1 << 31)
        else:
            packed_compression = packed_size & ~(1 << 31)

        packed_entry = struct.pack('<IIII', 
                                   self.entry_size(),
                                   self.offset(),
                                   unpacked_size,
                                   packed_compression
                                  )
        file.write(packed_entry + byte_pad_string(self.filename))

    @classmethod
    def read_from_file(cls, file):
        entry_size = struct.unpack_from('<I', file.read(4))[0]
        entry_data = file.read(entry_size - 4)

        # Deserialize the file entry
        offset, unpacked_size, packed_compression = struct.unpack_from('<III', entry_data)
        is_compressed = bool(packed_compression & (1 << 31))
        packed_size = packed_compression & 0x7FFFFFFF
        is_ghost_file = packed_size == 0 and unpacked_size == 0

        # Deserialize the filename
        filename = entry_data[12:].decode('ascii').rstrip('\x00')

        return cls(filename, offset, unpacked_size, packed_size, is_compressed, is_ghost_file)
    
    @classmethod
    def should_pack_compressed(cls, path):
        if cls.exclude_list == None:
            exclude_list_path = os.path.join(os.path.dirname(path), "..", "Graphics", "noncomp.txt")
            if os.path.isfile(exclude_list_path):
                with open(exclude_list_path) as f:
                    cls.exclude_list = {s.strip() for s in f.readlines()}
            else:
                cls.exclude_list = set()

        asset_name = os.path.splitext(os.path.basename(path))[0]
        if asset_name in cls.exclude_list:
            print(f"Skipping compression of {asset_name}")
            return False
        
        unpacked_size = os.path.getsize(path)
        try:
            packed_size = os.path.getsize(compressed(path))
        except subprocess.CalledProcessError:
            print(f"Failed to compress {os.path.basename(path)}")
            return False

        return packed_size < unpacked_size

    @classmethod
    def from_path(cls, path, no_compress):
        assert os.path.isfile(path)
        filename = os.path.basename(path)
        assert not filename.startswith(".")

        unpacked_size = os.path.getsize(path)
        should_pack_compressed = False if no_compress else cls.should_pack_compressed(path)
        packed_size = os.path.getsize(compressed(path)) if should_pack_compressed else unpacked_size

        return IndexFileEntry(filename,
                              None,  # The offset is calculated later
                              unpacked_size=unpacked_size,
                              packed_size=packed_size,
                              is_compressed=should_pack_compressed,
                              is_ghost_file=False)


class IndexSection:
    UNKNOWN_SECTION_NAME = "_UNKNOWN"
    SECTION_NAMES = [
        UNKNOWN_SECTION_NAME,
        'arenas',
        'components',
        'armour',
        'chassis',
        'drive',
        'locomotion',
        'power',
        'weapon',
        'fonts',
        'Graphics',
        'Language',
        'robots',
        'powertrain',
        'UI2',
        'objects',
        'Tournaments',
        'particles',
        'crowd'
    ]

    def __init__(self, section_name=None, entries=None):
        if section_name == self.UNKNOWN_SECTION_NAME:
            self._section_name = None
        else:
            self._section_name = section_name
        self.entries = entries or []
    
    def extracted_dir_name(self):
        return self._section_name or self.UNKNOWN_SECTION_NAME
    
    def section_name_padded(self):
        if self._section_name:
            return byte_pad_string(self._section_name)
        else:
            return b""
    
    def section_size(self):
        return 12 + len(self.section_name_padded())
    
    def unknown(self):
        return 0 if self.entries else 6

    def write_to_file(self, file):
        if not self._section_name:  # NOTE: number of entries and unknown are swapped for some reason
            packed_header = struct.pack('<III', self.section_size(), len(self.entries), self.unknown())
        else:
            packed_header = struct.pack('<III', self.section_size(), self.unknown(), len(self.entries))

        file.write(packed_header + self.section_name_padded())
        for entry in self.entries:
            entry.write_to_file(file)

    @classmethod
    def read_from_file(cls, file):
        # Deserialize the section header
        section_size = struct.unpack_from('<I', file.read(4))[0]
        section_data = file.read(section_size - 4)

        if section_size == 12:  # No name
            num_entries, unknown = struct.unpack_from('<II', section_data)
            section_name = None
        else:
            unknown, num_entries = struct.unpack_from('<II', section_data)
            if unknown:
                assert num_entries == 0 and unknown == 6
            # Deserialize the section name
            section_name = section_data[8:].decode('ascii').rstrip('\x00')

        # Deserialize each file entry
        entries = [IndexFileEntry.read_from_file(file) for _ in range(num_entries)]
        return cls(section_name=section_name, entries=entries)
    
    @classmethod
    def from_path(cls, asset_dir, section_name, no_compress):
        # Replicate the filename ordering of the original index file
        def filename_key(filename):
            filename = filename.lower().replace("_", "~")
            parts = re.findall(r'\d+|\D', filename)
            def number_if_possible(s):
                try:
                    return str(int(s)).zfill(10)
                except ValueError:
                    return s
            return tuple(number_if_possible(p) for p in parts)

        path = os.path.join(asset_dir, section_name)
        entries = []
        for filename in sorted(os.listdir(path), key=filename_key):
            filepath = os.path.join(path, filename)
            if not os.path.isfile(filepath) or filename.startswith("."):
                continue
            entries.append(IndexFileEntry.from_path(filepath, no_compress))
        return cls(section_name, entries)


class AssetIndex:
    # NOTE: with some custom files added, the game was crashing upon level load until I changed the
    # third byte from 2 (the original game's value) to 3. ¯\_(ツ)_/¯
    INDEX_HEADER = b"\xFC\xF5\x03\x00\x10\x00\x00\x00"

    def __init__(self, sections=None):
        self.sections = sections or []

    def write_to_file(self, file):
        # Serialize all sections
        file.write(self.INDEX_HEADER)
        for section in self.sections:
            section.write_to_file(file)

    @classmethod
    def read_from_file(cls, file):
        sections = []
        f.seek(len(cls.INDEX_HEADER))  # Skip the first 8 bytes (header)
        while file.peek(4):
            sections.append(IndexSection.read_from_file(file))

        return cls(sections=sections)
    
    @classmethod
    def from_asset_dir(cls, asset_dir: str, no_compress: bool):
        sections = []
        for section_name in IndexSection.SECTION_NAMES:
            section_path = os.path.join(asset_dir, section_name)
            assert os.path.isdir(section_path)
            sections.append(IndexSection.from_path(asset_dir, section_name, no_compress))

        return cls(sections)


class AssetData:
    def __init__(self, asset_index: AssetIndex, data_file_path: str):
        self.asset_index = asset_index
        self.data_file_path = data_file_path

    def extract_files(self, output_dir, only_section=None, only_filename=None):
        os.makedirs(output_dir, exist_ok=True)

        executable = "rnc_lib"
        executable_path = which(executable)

        tasks = []
        with open(self.data_file_path, "rb") as data_file:
            for section in self.asset_index.sections:
                section_dir = os.path.join(output_dir, section.extracted_dir_name())
                os.makedirs(section_dir, exist_ok=True)

                for entry in section.entries:
                    entry.update_from_data_file(data_file)
                    output_file_path = os.path.join(section_dir, entry.filename)

                    do_section = section.extracted_dir_name() == only_section or not only_section
                    do_file = entry.filename == only_filename or not only_filename
                    if do_section and do_file:
                        if entry.is_compressed() and not executable_path:
                            raise RuntimeError(f"{executable}.exe is missing from both the current"
                                               " directory and your PATH. Ensure RNC ProPackED is "
                                               "available in one of these locations.")

                        tasks.append((self.data_file_path, output_file_path, entry))

        # Run extraction tasks in parallel using multiprocessing
        with Pool(cpu_count()) as pool:
            pool.map(self._extract_file, tasks)

    def _extract_file(self, args):
        data_file_path, output_file_path, entry = args
        if entry.is_compressed():
            print(f"Decompressing {entry.filename}...")
            command = [
                'rnc_lib', 'u', data_file_path, output_file_path, f'-i={entry.offset():X}'
            ]
            subprocess.check_call(command, stdout=subprocess.DEVNULL)
            
            # Previously we'd extract the file verbatim for caching purposes, but it seems there's
            # a small discrepancy between the RNC compression used in the original game and that
            # available to us now. Both read/write just fine, but the size difference of the 
            # resulting files causes issues when repacking.
            # directly_extracted_path = to_compressed_path(output_file_path)
        else:
            with open(data_file_path, "rb") as data_file:
                data_file.seek(entry.offset())
                with open(output_file_path, "wb") as out_file:
                    out_file.write(data_file.read(entry.packed_data_size()))
        print(f'Extracted {entry.filename} to {output_file_path}.')

    def pack_files(self, asset_dir, ignore_cache):
        with open(self.data_file_path, 'wb') as data_file:
            current_offset = 0
            num_writes = 0
            for section in self.asset_index.sections:
                section_dir = os.path.join(asset_dir, section.extracted_dir_name())

                for entry in section.entries:
                    input_file_path = os.path.join(section_dir, entry.filename)

                    if entry.is_compressed():
                        input_file_path = compressed(input_file_path, ignore_cache)

                    with open(input_file_path, 'rb') as f:
                        packed_data = f.read()

                    current_offset = entry.write_packed_asset_to_data_file(packed_data, data_file, current_offset)
                    num_writes += 1

                    print(f'Packed {entry.filename} to {self.data_file_path} at offset {entry.offset():X}.')

        print(f"Packed {num_writes} files.")

    @classmethod
    def from_asset_dir(cls, asset_dir, data_file_path, no_compress):
        asset_index = AssetIndex.from_asset_dir(asset_dir, no_compress)
        return cls(asset_index, data_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-command to run")

    # Unpack command
    unpack_parser = subparsers.add_parser("unpack", help="Extract all assets from the data file using the index file.")
    unpack_parser.add_argument("index_file", help="Path to the index file (.idx)")
    unpack_parser.add_argument("data_file", help="Path to the data file (.dat)")
    unpack_parser.add_argument("output_dir", help="Directory to extract assets into")
    unpack_parser.add_argument("--section", required=False)
    unpack_parser.add_argument("--filename", required=False)

    # Pack command
    pack_parser = subparsers.add_parser("pack", help="Pack all assets from the given directory into the index and data files.")
    pack_parser.add_argument("asset_dir", help="Directory containing assets to pack")
    pack_parser.add_argument("index_file", help="Path to save the index file (.idx)")
    pack_parser.add_argument("data_file", help="Path to save the data file (.dat)")
    pack_parser.add_argument("--no-cache", required=False, action="store_true")
    pack_parser.add_argument("--no-compress",  required=False, action="store_true")

    args = parser.parse_args()

    if args.command == "unpack":
        with open(args.index_file, "rb") as f:
            asset_index = AssetIndex.read_from_file(f)
        asset_data = AssetData(asset_index, args.data_file)
        asset_data.extract_files(args.output_dir, only_section=args.section, only_filename=args.filename)
        print("Unpacking complete!")
    elif args.command == "pack":
        asset_data = AssetData.from_asset_dir(args.asset_dir, args.data_file, args.no_compress)
        asset_data.pack_files(args.asset_dir, ignore_cache=args.no_cache)
        with open(args.index_file, 'wb') as f:
            asset_data.asset_index.write_to_file(f)
        print("Packing complete!")
    else:
        raise argparse.ArgumentError("Unknown command!")