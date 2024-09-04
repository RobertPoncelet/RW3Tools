import argparse
import os
import struct
import json
import subprocess
from multiprocessing import Pool, cpu_count


class FileEntry:
    def __init__(self, file_entry=None, filename=None):
        self.file_entry = file_entry or (0, 0, 0, 0, 0)  # default 5-tuple
        self.filename = filename or ""
    
    def entry_size(self):
        return self.file_entry[0]
    
    def offset(self):
        return self.file_entry[1]

    def uncompressed_file_size(self):
        return self.file_entry[2]
    
    def compressed_file_size(self):
        return self.file_entry[3]
    
    def flags(self):
        return self.file_entry[4]

    def is_compressed(self):
        return bool(self.file_entry[5])

    def write_to_file(self):
        # Serialize the file entry and filename
        packed_entry = struct.pack('<IIIHBB', *self.file_entry)
        filename_bytes = self.filename.encode('ascii') + b'\x00'
        padding_length = (4 - (len(filename_bytes) % 4)) % 4
        filename_bytes += b'\x00' * padding_length
        return packed_entry + filename_bytes

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


class IndexSection:
    UNKNOWN_SECTION_NAME = "_UNKNOWN"
    SECTION_LIST = [
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

    def __init__(self, section_size=0, unknown=0, section_name=None, entries=None):
        self.section_size = section_size
        self.unknown = unknown
        self._section_name = section_name
        self.entries = entries or []
    
    def extracted_dir_name(self):
        return self._section_name or self.UNKNOWN_SECTION_NAME

    def write_to_file(self):
        if self.section_size == 12:  # No name
            packed_header = struct.pack('<III', self.section_size, len(self.entries), self.unknown)
            section_name_bytes = b""
        else:
            assert self._section_name
            packed_header = struct.pack('<III', self.section_size, self.unknown, len(self.entries))
            section_name_bytes = self._section_name.encode('ascii') + b'\x00'
            padding_length = (4 - (len(section_name_bytes) % 4)) % 4
            section_name_bytes += b'\x00' * padding_length

        # Serialize each file entry
        entry_data = b''.join(entry.write_to_file() for entry in self.entries)
        return packed_header + section_name_bytes + entry_data

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

        return cls(section_size=section_size, unknown=unknown, section_name=section_name, entries=entries), index


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
        for section_data in metadata:
            entries = [
                FileEntry(file_entry=entry['file_entry'], filename=entry['filename'])
                for entry in section_data['entries']
            ]
            section = IndexSection(
                section_size=section_data['type'],
                unknown=section_data['unknown'],
                section_name=section_data['name'],
                entries=entries
            )
            sections.append(section)

        return cls(sections)


class AssetData:
    def __init__(self, asset_index: AssetIndex, dat_file_path: str):
        self.asset_index = asset_index
        self.dat_file_path = dat_file_path

    def extract_files(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        sections_metadata = []
        tasks = []

        for section in self.asset_index.sections:
            section_dir = os.path.join(output_dir, section.extracted_dir_name())
            os.makedirs(section_dir, exist_ok=True)

            entries_metadata = []

            for entry in section.entries:
                filename = entry.filename
                output_file_path = os.path.join(section_dir, filename)

                tasks.append((self.dat_file_path, output_file_path, entry))

                # Collect metadata needed to reconstruct the idx file
                entries_metadata.append({
                    'filename': filename,
                    'file_entry': entry.file_entry
                })
            
            sections_metadata.append({
                'name': section.extracted_dir_name(),
                'type': section.section_size,
                'unknown': section.unknown,
                'entries': entries_metadata
            })

        # Save metadata to JSON
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as json_file:
            json.dump(sections_metadata, json_file, indent=4)

        # Run extraction tasks in parallel using multiprocessing
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

            for entry in section.entries:
                input_file_path = os.path.join(section_dir, entry.filename)

                temp_dat_path = input_file_path + '.dat'
                command = [
                    'rnc_lib.exe', 'p', input_file_path, temp_dat_path, '-m=1'
                ]
                subprocess.check_call(command, stdout=subprocess.DEVNULL)

                with open(temp_dat_path, 'rb') as temp_file:
                    packed_data = temp_file.read()

                with open(self.dat_file_path, 'r+b') as dat_file:
                    dat_file.seek(entry.offset())
                    dat_file.write(packed_data)

                os.remove(temp_dat_path)

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

    # Pack command
    pack_parser = subparsers.add_parser("pack", help="Pack all assets from the given directory into the index and data files.")
    pack_parser.add_argument("asset_dir", help="Directory containing assets to pack")
    pack_parser.add_argument("index_file", help="Path to save the index file (.idx)")
    pack_parser.add_argument("data_file", help="Path to save the data file (.dat)")

    args = parser.parse_args()

    if args.command == "unpack":
        asset_index = AssetIndex.read_from_file(args.index_file)
        asset_data = AssetData(asset_index, args.data_file)
        asset_data.extract_files(args.output_dir)
    elif args.command == "pack":
        asset_data = AssetData.from_asset_dir(args.asset_dir, args.data_file)
        asset_data.pack_files()
        with open(args.index_file, 'wb') as f:
            f.write(asset_data.asset_index.write_to_file())
    else:
        raise argparse.ArgumentError("Unknown command!")