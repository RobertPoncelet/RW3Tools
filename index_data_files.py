import struct, os, subprocess, json


def read_string(data, index):
    string_end = data.index(b'\x00', index) + 1
    string = data[index:string_end].decode('ascii').rstrip('\x00')
    index = (string_end + 3) & ~3  # Move to the next 4-byte boundary
    return string, index


UNKNOWN_NAME = "_UNKNOWN"
WEIRD_SECTION_TYPE = 0x0C


def parse_idx_file(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()

    idx_data = {}
    index = 8  # Skip the first 8 bytes (header)

    while index < len(data):
        # Parse the section header
        section_type, unknown, num_entries = struct.unpack_from('<III', data, index)
        index += 12  # Move past the three numbers

        if section_type == WEIRD_SECTION_TYPE:
            # Weird special section type
            temp = num_entries
            num_entries = unknown  # The number of entries seems to be in the middle this time
            unknown = temp
            section_name = UNKNOWN_NAME
        else:
            # Parse the section name
            section_name_end = data.index(b'\x00', index) + 1
            section_name = data[index:section_name_end].decode('ascii').rstrip('\x00')
            index = (section_name_end + 3) & ~3  # Move to the next 4-byte boundary
        print("Parsing section", section_name)

        # Initialize the section in our dictionary
        idx_data[section_name] = {
            'type': section_type,
            'unknown': unknown,
            'entries': []
        }

        # Parse the file entries in the section
        for _ in range(num_entries):
            file_entry = struct.unpack_from('<IIII', data, index)
            index += 16  # Move past the four numbers

            # Parse the filename
            filename_end = data.index(b'\x00', index) + 1
            filename = data[index:filename_end].decode('ascii').rstrip('\x00')
            index = (filename_end + 3) & ~3  # Move to the next 4-byte boundary

            # Store the file entry data
            idx_data[section_name]['entries'].append({
                'file_entry': file_entry,
                'filename': filename,
                'offset': file_entry[1]  # The second number is the offset in the .dat file
            })

    return idx_data


def extract_files_from_dat(dat_file_path, idx_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    metadata = {}

    for section_name, section_data in idx_data.items():
        section_dir = os.path.join(output_dir, section_name)
        os.makedirs(section_dir, exist_ok=True)

        metadata[section_name] = {
            'type': section_data['type'],
            'unknown': section_data['unknown'],
            'entries': []
        }

        for entry in section_data['entries']:
            offset = entry['offset']
            filename = entry['filename']
            output_file_path = os.path.join(section_dir, filename)

            # Extract the file using rnc_lib
            command = [
                'rnc_lib.exe', 'u', dat_file_path, output_file_path, f'-i={offset:X}'
            ]
            subprocess.run(command)

            # Collect metadata needed to reconstruct the idx file
            metadata[section_name]['entries'].append({
                'filename': filename,
                'file_entry': entry['file_entry']
            })

            print(f'Extracted {filename} to {output_file_path}.')

    # Save metadata to JSON
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as json_file:
        json.dump(metadata, json_file, indent=4)


def create_idx_file(idx_file_path, metadata, output_dir):
    with open(idx_file_path, 'wb') as f:
        # Write the 8-byte header
        f.write(b'\xFC\xF5\x02\x00\x10\x00\x00\x00')

        for section_name, section_data in metadata.items():
            # Write the section header
            section_type = section_data['type']
            unknown = section_data['unknown']
            num_entries = len(section_data['entries'])

            # Handle the weird section type
            if section_type == WEIRD_SECTION_TYPE:
                f.write(struct.pack('<III', section_type, num_entries, unknown))
            else:
                f.write(struct.pack('<III', section_type, unknown, num_entries))
                # Write the section name, null-terminated and padded to 4-byte boundary
                padded_section_name = section_name.encode('ascii') + b'\x00'
                f.write(padded_section_name)
                padding_length = (4 - (len(padded_section_name) % 4)) % 4
                f.write(b'\x00' * padding_length)

            for entry in section_data['entries']:
                # Write the file entry
                original_entry = entry['file_entry']
                # Pack the four numbers
                f.write(struct.pack('<IIII', *original_entry))

                # Write the filename, null-terminated and padded to 4-byte boundary
                filename = entry['filename'].encode('ascii') + b'\x00'
                f.write(filename)
                padding_length = (4 - (len(filename) % 4)) % 4
                f.write(b'\x00' * padding_length)


def pack_files_to_dat(dat_file_path, metadata, output_dir):
    with open(dat_file_path, 'wb') as dat_file:
        for section_name, section_data in metadata.items():
            section_dir = os.path.join(output_dir, section_name)

            for entry in section_data['entries']:
                filename = entry['filename']
                input_file_path = os.path.join(section_dir, filename)
                offset = entry['file_entry'][1]

                # Pack the file using rnc_lib.exe
                temp_dat_path = input_file_path + '.dat'
                command = [
                    'rnc_lib.exe', 'p', input_file_path, temp_dat_path, '-m=1'
                ]
                subprocess.run(command)

                # Move the packed data to the correct offset in the .dat file
                with open(temp_dat_path, 'rb') as temp_file:
                    packed_data = temp_file.read()

                # Seek to the correct offset
                dat_file.seek(offset)
                dat_file.write(packed_data)

                # Remove the temporary .dat file
                os.remove(temp_dat_path)

                print(f'Packed {filename} to {dat_file_path} at offset {offset:X}.')


BASE_DIR = "F:/Google Drive/RW3/assets/"
INDEX_FILE = "lumpy.idx"
DATA_FILE = "lumpy.dat"


if __name__ == "__main__":
    idx_data = parse_idx_file(os.path.join(BASE_DIR, INDEX_FILE))
    print(len(idx_data), "sections:", ", ".join(idx_data.keys()))
    os.chdir(BASE_DIR)
    extract_files_from_dat(os.path.join(BASE_DIR, DATA_FILE), idx_data, os.path.join(BASE_DIR, "extracted"))