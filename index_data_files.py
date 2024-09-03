import argparse
import os
import struct
import json
import subprocess
from multiprocessing import Pool, cpu_count


UNKNOWN_NAME = "_UNKNOWN"


class FileEntry:
    def __init__(self, file_entry=None, filename=None):
        self.file_entry = file_entry or (0, 0, 0, 0, 0)  # default 5-tuple
        self.filename = filename or ""
        self.offset = self.file_entry[1]  # Second number is the offset
        self.compressed = bool(self.file_entry[4])

    def serialize(self):
        # Serialize the file entry and filename
        packed_entry = struct.pack('<IIIHH', *self.file_entry)
        filename_bytes = self.filename.encode('ascii') + b'\x00'
        # TODO: surely the padding can't be right if we don't know where we are in the file
        padding_length = (4 - (len(filename_bytes) % 4)) % 4
        filename_bytes += b'\x00' * padding_length
        return packed_entry + filename_bytes

    @classmethod
    def read_from_file(cls, data, index):
        # Deserialize the file entry
        file_entry = struct.unpack_from('<IIIHH', data, index)
        index += 16  # Move past the numbers

        # Deserialize the filename
        filename_end = data.index(b'\x00', index) + 1
        filename = data[index:filename_end].decode('ascii').rstrip('\x00')
        index = (filename_end + 3) & ~3  # Move to the next 4-byte boundary

        return cls(file_entry=file_entry, filename=filename), index


class DataSection:
    WEIRD_SECTION_TYPE = 0x0C

    def __init__(self, section_type=0, unknown=0, section_name="", entries=None):
        self.section_type = section_type
        self.unknown = unknown
        self.section_name = section_name
        self.entries = entries or []

    def serialize(self):
        if self.section_type == self.WEIRD_SECTION_TYPE:
            packed_header = struct.pack('<III', self.section_type, len(self.entries), self.unknown)
            section_name_bytes = b""
        else:
            packed_header = struct.pack('<III', self.section_type, self.unknown, len(self.entries))
            section_name_bytes = self.section_name.encode('ascii') + b'\x00'
            padding_length = (4 - (len(section_name_bytes) % 4)) % 4
            section_name_bytes += b'\x00' * padding_length

        # Serialize each file entry
        entry_data = b''.join(entry.serialize() for entry in self.entries)
        return packed_header + section_name_bytes + entry_data

    @classmethod
    def read_from_file(cls, data, index):
        # Deserialize the section header
        section_type = struct.unpack_from('<I', data, index)
        index += 4

        if section_type == cls.WEIRD_SECTION_TYPE:
            num_entries, unknown = struct.unpack_from('<II', data, index)
            index += 8
        else:
            unknown, num_entries = struct.unpack_from('<II', data, index)
            index += 8

            # Deserialize the section name
            section_name_end = data.index(b'\x00', index) + 1
            section_name = data[index:section_name_end].decode('ascii').rstrip('\x00')
            index = (section_name_end + 3) & ~3  # Move to the next 4-byte boundary

        # Deserialize each file entry
        entries = []
        for _ in range(num_entries):
            entry, index = FileEntry.read_from_file(data, index)
            entries.append(entry)

        return cls(section_type=section_type, unknown=unknown, section_name=section_name, entries=entries), index


class IndexFile:
    INDEX_HEADER = b'\xFC\xF5\x02\x00\x10\x00\x00\x00'

    def __init__(self, sections=None):
        self.sections = sections or []

    def serialize(self):
        # Serialize all sections
        return self.INDEX_HEADER + b''.join(section.serialize() for section in self.sections)

    @classmethod
    def read_from_file(cls, file_path):
        with open(file_path, 'rb') as f:
            data = f.read()

        sections = []
        index = len(cls.INDEX_HEADER)  # Skip the first 8 bytes (header)
        while index < len(data):
            section, index = DataSection.read_from_file(data, index)
            sections.append(section)

        return cls(sections=sections)


class DataFile:
    def __init__(self, idx_file, dat_file_path):
        self.idx_file = idx_file
        self.dat_file_path = dat_file_path

    def extract_files(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        metadata = {}

        tasks = []

        for section in self.idx_file.sections:
            section_dir = os.path.join(output_dir, section.section_name)
            os.makedirs(section_dir, exist_ok=True)

            metadata[section.section_name] = {
                'type': section.section_type,
                'unknown': section.unknown,
                'entries': []
            }

            for entry in section.entries:
                offset = entry.offset
                filename = entry.filename
                output_file_path = os.path.join(section_dir, filename)

                tasks.append((self.dat_file_path, output_file_path, offset))

                # Collect metadata needed to reconstruct the idx file
                metadata[section.section_name]['entries'].append({
                    'filename': filename,
                    'file_entry': entry.file_entry
                })

        # Run extraction tasks in parallel using multiprocessing
        with Pool(cpu_count()) as pool:
            pool.map(self._extract_file, tasks)

        # Save metadata to JSON
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as json_file:
            json.dump(metadata, json_file, indent=4)

    def _extract_file(self, args):
        dat_file_path, output_file_path, offset = args
        command = [
            'rnc_lib.exe', 'u', dat_file_path, output_file_path, f'-i={offset:X}'
        ]
        subprocess.check_call(command, check=True)
        print(f'Extracted {os.path.basename(output_file_path)} to {output_file_path}.')

    def pack_files(self, output_dir):
        with open(os.path.join(output_dir, 'metadata.json'), 'r') as json_file:
            metadata = json.load(json_file)

        for section_name, section_data in metadata.items():
            section_dir = os.path.join(output_dir, section_name)

            for entry_data in section_data['entries']:
                filename = entry_data['filename']
                input_file_path = os.path.join(section_dir, filename)
                offset = entry_data['file_entry'][1]

                temp_dat_path = input_file_path + '.dat'
                command = [
                    'rnc_lib.exe', 'p', input_file_path, temp_dat_path, '-m=1'
                ]
                subprocess.check_call(command, check=True)

                with open(temp_dat_path, 'rb') as temp_file:
                    packed_data = temp_file.read()

                with open(self.dat_file_path, 'r+b') as dat_file:
                    dat_file.seek(offset)
                    dat_file.write(packed_data)

                os.remove(temp_dat_path)

                print(f'Packed {filename} to {self.dat_file_path} at offset {offset:X}.')

    def create_idx_file(self, output_path):
        with open(os.path.join(output_path, 'metadata.json'), 'r') as json_file:
            metadata = json.load(json_file)

        sections = []
        for section_name, section_data in metadata.items():
            entries = [
                FileEntry(file_entry=entry['file_entry'], filename=entry['filename'])
                for entry in section_data['entries']
            ]
            section = DataSection(
                section_type=section_data['type'],
                unknown=section_data['unknown'],
                section_name=section_name,
                entries=entries
            )
            sections.append(section)

        idx_file = IndexFile(sections)
        with open(output_path, 'wb') as f:
            f.write(idx_file.serialize())


# BASE_DIR = "F:/Google Drive/RW3/assets/"
# INDEX_FILE = "lumpy.idx"
# DATA_FILE = "lumpy.dat"


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
        process_files('read', args.index_file, args.data_file, args.output_dir)
    elif args.command == "pack":
        process_files('write', args.index_file, args.data_file, args.asset_dir)
    else:
        raise argparse.ArgumentError("Unknown command!")