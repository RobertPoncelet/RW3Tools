import argparse
import json
import os
import struct
import re
import subprocess
from collections import defaultdict
from multiprocessing import Pool, cpu_count


def byte_pad_string(string):
    string_bytes = string.encode('ascii') + b'\x00'
    padding_length = (4 - (len(string_bytes) % 4)) % 4
    return string_bytes + b'\x00' * padding_length


def to_compressed_path(path):
    return os.path.join(os.path.dirname(path), f".{os.path.basename(path)}.dat")


def compressed(path):
    compressed_path = to_compressed_path(path)
    if os.path.isfile(compressed_path) and os.path.getmtime(path) < os.path.getmtime(compressed_path):
        return compressed_path

    print(f"Compressing {path}...")
    command = [
        'rnc_lib.exe', 'p', path, compressed_path, '-m=1'
    ]
    subprocess.check_call(command, stdout=subprocess.DEVNULL)
    return compressed_path


class IndexFileEntry:
    def __init__(self, filename, offset, uncompressed_size, compressed_size, is_compressed, is_ghost_file):
        self.filename = filename
        self._offset = offset
        self._uncompressed_size = uncompressed_size
        self._compressed_size = compressed_size
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
            self._uncompressed_size = struct.unpack_from('>I', data_file.read(4))[0]
            print(f"Real uncompressed size of {self.filename}: {self._uncompressed_size}")
            self._compressed_size = struct.unpack_from('>I', data_file.read(4))[0]
            self._compressed_size += 18  # Account for the RNC header
            print(f"Real compressed size of {self.filename}: {self._compressed_size}")
        else:
            assert not self.is_compressed()
            # Fuck knows
            self._compressed_size = self._uncompressed_size = 0x800

    def write_packed_asset_to_data_file(self, packed_data, data_file, current_offset):
        # This is mainly so we can update our offset
        self._offset = current_offset
        assert len(packed_data) == self.compressed_file_size()
        data_file.write(packed_data)
        CHUNK_SIZE = 4  # Use 0x80 to get the original 2KB optimised for CD reading
        padding_length = (CHUNK_SIZE - (len(packed_data) % CHUNK_SIZE)) % CHUNK_SIZE
        data_file.write(b'\x00' * padding_length)
        current_offset = self.offset() + self.compressed_file_size() + padding_length
        return current_offset

    def is_ghost_file(self):
        return self._is_ghost_file
    
    def _file_size(self, compressed):
        size = self._compressed_size if compressed else self._uncompressed_size
        if not size:
            raise ValueError("Size can't be 0! Should probably update this entry from the data file first.")
        return size

    def uncompressed_file_size(self):
        return self._file_size(False)
    
    def compressed_file_size(self):
        return self._file_size(True)

    def is_compressed(self):
        return self._is_compressed

    def write_to_file(self, file):
        # Serialize the file entry and filename
        if self.is_ghost_file():
            uncompressed_size = packed_compression = 0
        else:
            uncompressed_size = self.uncompressed_file_size()
            packed_compression = self.compressed_file_size()

        if self.is_compressed():
            packed_compression |= (1 << 31)
        else:
            packed_compression &= ~(1 << 31)

        packed_entry = struct.pack('<IIII', 
                                   self.entry_size(),
                                   self.offset(),
                                   uncompressed_size,
                                   packed_compression
                                  )
        file.write(packed_entry + byte_pad_string(self.filename))

    @classmethod
    def read_from_file(cls, file):
        entry_size = struct.unpack_from('<I', file.read(4))[0]
        entry_data = file.read(entry_size - 4)

        # Deserialize the file entry
        offset, uncompressed_size, packed_compression = struct.unpack_from('<III', entry_data)
        is_compressed = bool(packed_compression & (1 << 31))
        compressed_size = packed_compression & 0x7FFFFFFF
        is_ghost_file = compressed_size == 0 and uncompressed_size == 0

        # Deserialize the filename
        filename = entry_data[12:].decode('ascii').rstrip('\x00')

        return cls(filename, offset, uncompressed_size, compressed_size, is_compressed, is_ghost_file)

    @classmethod
    def from_path(cls, path, entry_metadata):
        assert os.path.isfile(path)
        filename = os.path.basename(path)
        assert not filename.startswith(".")

        uncompressed_size = os.path.getsize(path)
        try:
            compressed_size = os.path.getsize(compressed(path))
        except subprocess.CalledProcessError:
            print(f"Failed to compress {filename}")
            compressed_size = uncompressed_size
        should_compress = compressed_size < uncompressed_size
        ghost_file = (entry_metadata or {}).get("is_ghost_file", False)
        return IndexFileEntry(filename,
                              None,  # The offset is calculated later
                              uncompressed_size=os.path.getsize(path),
                              compressed_size=compressed_size,
                              is_compressed=should_compress,
                              is_ghost_file=ghost_file)


class IndexSection:
    UNKNOWN_SECTION_NAME = "_UNKNOWN"

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
    def from_path(cls, path, section_metadata):
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

        entries = []
        for filename in sorted(os.listdir(path), key=filename_key):
            filepath = os.path.join(path, filename)
            if not os.path.isfile(filepath) or filename.startswith("."):
                continue
            entries.append(IndexFileEntry.from_path(filepath, section_metadata["entries"].get(filename)))
        return cls(section_metadata["name"], entries)


class AssetIndex:
    INDEX_HEADER = b'\xFC\xF5\x02\x00\x10\x00\x00\x00'

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
    def from_asset_dir(cls, asset_dir):
        with open(os.path.join(asset_dir, 'metadata.json'), 'r') as json_file:
            metadata = json.load(json_file)

        sections = []
        for section_metadata in metadata:
            section_path = os.path.join(asset_dir, section_metadata["name"])
            assert os.path.isdir(section_path)
            sections.append(IndexSection.from_path(section_path, section_metadata))

        return cls(sections)


class AssetData:
    def __init__(self, asset_index: AssetIndex, data_file_path: str):
        self.asset_index = asset_index
        self.data_file_path = data_file_path

    def extract_files(self, output_dir, only_section=None, metadata_only=False):
        os.makedirs(output_dir, exist_ok=True)

        sections_metadata = []
        tasks = []

        with open(self.data_file_path, "rb") as data_file:
            for section in self.asset_index.sections:
                section_dir = os.path.join(output_dir, section.extracted_dir_name())
                os.makedirs(section_dir, exist_ok=True)

                entries_metadata = defaultdict(dict)

                for entry in section.entries:
                    entry.update_from_data_file(data_file)
                    output_file_path = os.path.join(section_dir, entry.filename)

                    if section.extracted_dir_name() == only_section or not only_section:
                        tasks.append((self.data_file_path, output_file_path, entry))

                    # Collect metadata needed to reconstruct the idx file
                    if entry.is_ghost_file():
                        entries_metadata[entry.filename]["is_ghost_file"] = True
                
                section_item = {
                    'name': section.extracted_dir_name(),
                    'entries': entries_metadata
                }
                sections_metadata.append(section_item)

        # Save metadata to JSON
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as json_file:
            json.dump(sections_metadata, json_file, indent=4)

        # Run extraction tasks in parallel using multiprocessing
        if not metadata_only:
            with Pool(cpu_count()) as pool:
                pool.map(self._extract_file, tasks)

    def _extract_file(self, args):
        data_file_path, output_file_path, entry = args
        if entry.is_compressed():
            print(f"Decompressing {entry.filename}...")
            command = [
                'rnc_lib.exe', 'u', data_file_path, output_file_path, f'-i={entry.offset():X}'
            ]
            subprocess.check_call(command, stdout=subprocess.DEVNULL)
            # Extract the file verbatim for caching
            directly_extracted_path = to_compressed_path(output_file_path)
        else:
            directly_extracted_path = output_file_path

        with open(data_file_path, "rb") as data_file:
            data_file.seek(entry.offset())
            with open(directly_extracted_path, "wb") as out_file:
                out_file.write(data_file.read(entry.compressed_file_size()))
        print(f'Extracted {entry.filename} to {output_file_path}.')

    def pack_files(self, asset_dir):
        with open(self.data_file_path, 'wb') as data_file:
            current_offset = 0
            for section in self.asset_index.sections:
                section_dir = os.path.join(asset_dir, section.extracted_dir_name())

                for entry in section.entries:
                    input_file_path = os.path.join(section_dir, entry.filename)

                    if entry.is_compressed():
                        input_file_path = compressed(input_file_path)

                    with open(input_file_path, 'rb') as f:
                        packed_data = f.read()

                    current_offset = entry.write_packed_asset_to_data_file(packed_data, data_file, current_offset)

                    print(f'Packed {entry.filename} to {self.data_file_path} at offset {entry.offset():X}.')

    @classmethod
    def from_asset_dir(cls, asset_dir, data_file_path):
        asset_index = AssetIndex.from_asset_dir(asset_dir)
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

    # Pack command
    pack_parser = subparsers.add_parser("pack", help="Pack all assets from the given directory into the index and data files.")
    pack_parser.add_argument("asset_dir", help="Directory containing assets to pack")
    pack_parser.add_argument("index_file", help="Path to save the index file (.idx)")
    pack_parser.add_argument("data_file", help="Path to save the data file (.dat)")

    args = parser.parse_args()

    if args.command == "unpack":
        with open(args.index_file, "rb") as f:
            asset_index = AssetIndex.read_from_file(f)
        asset_data = AssetData(asset_index, args.data_file)
        asset_data.extract_files(args.output_dir, only_section=args.section)
        print("Unpacking complete!")
    elif args.command == "pack":
        asset_data = AssetData.from_asset_dir(args.asset_dir, args.data_file)
        asset_data.pack_files(args.asset_dir)
        with open(args.index_file, 'wb') as f:
            asset_data.asset_index.write_to_file(f)
        print("Packing complete!")
    else:
        raise argparse.ArgumentError("Unknown command!")