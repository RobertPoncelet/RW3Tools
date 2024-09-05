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


class FileEntry:
    def __init__(self, filename, file_entry=None):
        self.filename = filename
        self.file_entry = file_entry or [0] * 6
        self.real_uncompressed_size = None
        self.real_compressed_size = None
        self._cached_compressed_path = None

    def __del__(self):
        if self._cached_compressed_path:
            pass#os.remove(self._cached_compressed_path)
    
    def entry_size(self):
        return 16 + len(byte_pad_string(self.filename))
    
    def offset(self):
        return self.file_entry[1]
    
    def update_from_data_file(self, data_file):
        # For some reason, certain files are marked as size 0 in the index, even though they exist
        # in the data file. In these cases, we need to check the data file itself.
        if not self.is_ghost_file():
            return

        data_file.seek(self.offset())
        header = data_file.read(4)
        if header == b"RNC\x01":  # RNC compression
            self.real_uncompressed_size = struct.unpack_from('>I', data_file.read(4))[0]
            print(f"Real uncompressed size of {self.filename}: {self.real_uncompressed_size}")
            self.real_compressed_size = struct.unpack_from('>I', data_file.read(4))[0]
            print(f"Real compressed size of {self.filename}: {self.real_compressed_size}")
        else:
            # Fuck knows
            self.real_compressed_size = self.real_uncompressed_size = 0x800

    def is_ghost_file(self):
        return self.file_entry[2] == 0 or self.file_entry[3] == 0
    
    def _file_size(self, compressed):
        size = self.file_entry[3 if compressed else 2]
        if size == 0:
            size = self.real_compressed_size if compressed else self.real_uncompressed_size
        if not size:
            raise ValueError("Size can't be 0! Should probably update this entry from the data file first.")
        return size

    def uncompressed_file_size(self):
        return self._file_size(False)
    
    def compressed_file_size(self):
        return self._file_size(True)
    
    def flags(self):
        return self.file_entry[4]

    def is_compressed(self):
        return bool(self.file_entry[5]) or self.compressed_file_size() != self.uncompressed_file_size()

    def write_to_file(self):
        # Serialize the file entry and filename
        print(self.filename, self.file_entry)
        packed_entry = struct.pack('<IIIHBB', *self.file_entry)
        return packed_entry + byte_pad_string(self.filename)

    @classmethod
    def read_from_file(cls, data, index):
        entry_size = struct.unpack_from('<I', data, index)[0]
        entry_data = data[index:index+entry_size]
        index += entry_size

        # Deserialize the file entry
        file_entry = list(struct.unpack_from('<IIIHBB', entry_data))

        # Deserialize the filename
        filename = entry_data[16:].decode('ascii').rstrip('\x00')

        return cls(file_entry=file_entry, filename=filename), index
    
    @staticmethod
    def compressed_path(path):
        return os.path.join(os.path.dirname(path), f".{os.path.basename(path)}.dat")

    @staticmethod
    def compress_file(path):
        temp_dat_path = FileEntry.compressed_path(path)
        # TODO: check file date etc.
        if os.path.isfile(temp_dat_path):
            return temp_dat_path
        command = [
            'rnc_lib.exe', 'p', path, temp_dat_path, '-m=1'
        ]
        subprocess.check_call(command, stdout=subprocess.DEVNULL)
        return temp_dat_path

    @classmethod
    def from_path(cls, path, entry_metadata):
        assert os.path.isfile(path)
        filename = os.path.basename(path)
        assert not filename.startswith(".")

        # TODO: this "file_entry" stuff needs refactoring
        file_entry = [
            16 + len(byte_pad_string(filename)),  # Entry size
            entry_metadata["offset"],  # TODO: calculate the offset as needed when writing the data file
            os.path.getsize(path),  # Uncompressed size
            None, #cls.compressed_size_from_path(path, entry_metadata),  # Compressed size
            None, #entry_metadata.get("flags", 0),  # Flags(?)
            0x80 if entry_metadata.get("is_compressed", True) else 0x00  # Is compressed
        ]

        entry = FileEntry(filename, file_entry)
    
        # Let's hack in the compressed size for now
        if entry_metadata.get("is_compressed", True):
            entry._cached_compressed_path = cls.compress_file(path)
            compressed_size = os.path.getsize(entry._cached_compressed_path)
        else:
            compressed_size = os.path.getsize(path)
        entry.file_entry[3] = compressed_size & 0xFFFF
        entry.file_entry[4] = compressed_size // 0xFFFF

        if entry_metadata.get("is_ghost_file", False):
            entry.real_uncompressed_size = file_entry[2]
            entry.real_compressed_size = file_entry[3]
            entry.file_entry[2] = entry.file_entry[3] = 0
        print("Created FileEntry for", path)
        return entry


class IndexSection:
    UNKNOWN_SECTION_NAME = "_UNKNOWN"

    def __init__(self, unknown=0, section_name=None, entries=None):
        self.unknown = unknown
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

    def write_to_file(self):
        if not self._section_name:  # NOTE: number of entries and unknown are swapped for some reason
            packed_header = struct.pack('<III', self.section_size(), len(self.entries), self.unknown)
        else:
            packed_header = struct.pack('<III', self.section_size(), self.unknown, len(self.entries))

        # Serialize each file entry
        entry_data = b''.join(entry.write_to_file() for entry in self.entries)
        return packed_header + self.section_name_padded() + entry_data

    @classmethod
    def read_from_file(cls, data, index):
        # Deserialize the section header
        section_size = struct.unpack_from('<I', data, index)[0]
        section_data = data[index:index+section_size]
        index += section_size

        if section_size == 12:  # No name
            _, num_entries, unknown = struct.unpack_from('<III', section_data)
            section_name = None
        else:
            _, unknown, num_entries = struct.unpack_from('<III', section_data)
            if unknown:
                assert num_entries == 0
            # Deserialize the section name
            section_name = section_data[12:].decode('ascii').rstrip('\x00')

        # Deserialize each file entry
        entries = []
        for _ in range(num_entries):
            entry, index = FileEntry.read_from_file(data, index)
            entries.append(entry)

        return cls(unknown=unknown, section_name=section_name, entries=entries), index
    
    @classmethod
    def from_path(cls, path, section_metadata):
        # Replicate the filename ordering of the original index file
        def filename_key(filename):
            filename = filename.lower().replace("_", "z")
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
            entries.append(FileEntry.from_path(filepath, section_metadata["entries"].get(filename)))
        return cls(section_metadata.get("unknown", 0), section_metadata["name"], entries)


class AssetIndex:
    INDEX_HEADER = b'\xFC\xF5\x02\x00\x10\x00\x00\x00'

    def __init__(self, sections=None):
        self.sections = sections or []

    def write_to_file(self):
        # Serialize all sections
        return self.INDEX_HEADER + b''.join(section.write_to_file() for section in self.sections)

    @classmethod
    def read_from_file(cls, file_path):
        with open(file_path, 'rb') as f:
            data = f.read()

        sections = []
        index = len(cls.INDEX_HEADER)  # Skip the first 8 bytes (header)
        while index < len(data):
            section, index = IndexSection.read_from_file(data, index)
            sections.append(section)

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
    def __init__(self, asset_index: AssetIndex, dat_file_path: str):
        self.asset_index = asset_index
        self.dat_file_path = dat_file_path

    def extract_files(self, output_dir, only_section=None, metadata_only=False):
        os.makedirs(output_dir, exist_ok=True)

        sections_metadata = []
        tasks = []

        with open(self.dat_file_path, "rb") as data_file:
            for section in self.asset_index.sections:
                section_dir = os.path.join(output_dir, section.extracted_dir_name())
                os.makedirs(section_dir, exist_ok=True)

                entries_metadata = defaultdict(dict)

                for entry in section.entries:
                    entry.update_from_data_file(data_file)
                    output_file_path = os.path.join(section_dir, entry.filename)

                    if section.extracted_dir_name() == only_section or not only_section:
                        tasks.append((self.dat_file_path, output_file_path, entry))

                    # Collect metadata needed to reconstruct the idx file
                    if not entry.is_compressed():
                        entries_metadata[entry.filename]["is_compressed"] = False
                    if entry.is_ghost_file():
                        entries_metadata[entry.filename]["is_ghost_file"] = True
                    if entry.flags():
                        entries_metadata[entry.filename]["flags"] = entry.flags()

                    # TODO: calculate the offset as needed when writing the data file
                    entries_metadata[entry.filename]["offset"] = entry.offset()
                
                section_item = {
                    'name': section.extracted_dir_name(),
                    'entries': entries_metadata
                }
                if section.unknown:
                    section_item["unknown"] = section.unknown
                sections_metadata.append(section_item)

        # Save metadata to JSON
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as json_file:
            json.dump(sections_metadata, json_file, indent=4)

        # Run extraction tasks in parallel using multiprocessing
        if not metadata_only:
            with Pool(cpu_count()) as pool:
                pool.map(self._extract_file, tasks)

    def _extract_file(self, args):
        dat_file_path, output_file_path, entry = args
        if entry.is_compressed():
            command = [
                'rnc_lib.exe', 'u', dat_file_path, output_file_path, f'-i={entry.offset():X}'
            ]
            subprocess.check_call(command, stdout=subprocess.DEVNULL)
        else:
            print(entry.filename, "is not compressed.")
            with open(dat_file_path, "rb") as dat_file:
                dat_file.seek(entry.offset())
                with open(output_file_path, "wb") as out_file:
                    out_file.write(dat_file.read(entry.uncompressed_file_size()))
        print(f'Extracted {os.path.basename(output_file_path)} to {output_file_path}.')

    def pack_files(self, asset_dir):
        for section in self.asset_index.sections:
            section_dir = os.path.join(asset_dir, section.extracted_dir_name())

            with open(self.dat_file_path, 'wb') as dat_file:
                for entry in section.entries:
                    input_file_path = os.path.join(section_dir, entry.filename)

                    if entry.is_compressed():
                        temp_dat_path = FileEntry.compress_file(input_file_path)
                        with open(temp_dat_path, 'rb') as temp_file:
                            packed_data = temp_file.read()
                        #os.remove(temp_dat_path)
                        entry._cached_compressed_path = temp_dat_path
                    else:
                        with open(input_file_path, 'rb') as temp_file:
                            packed_data = temp_file.read()

                    dat_file.seek(entry.offset())
                    dat_file.write(packed_data)

                    print(f'Packed {entry.filename} to {self.dat_file_path} at offset {entry.offset():X}.')

    @classmethod
    def from_asset_dir(cls, asset_dir, dat_file_path):
        asset_index = AssetIndex.from_asset_dir(asset_dir)
        return cls(asset_index, dat_file_path)


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
        asset_index = AssetIndex.read_from_file(args.index_file)
        asset_data = AssetData(asset_index, args.data_file)
        asset_data.extract_files(args.output_dir, only_section=args.section)
    elif args.command == "pack":
        asset_data = AssetData.from_asset_dir(args.asset_dir, args.data_file)
        with open(args.index_file, 'wb') as f:
            f.write(asset_data.asset_index.write_to_file())
        asset_data.pack_files(args.asset_dir)
    else:
        raise argparse.ArgumentError("Unknown command!")