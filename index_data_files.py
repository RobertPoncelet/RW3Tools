import struct, os, subprocess


def read_string(data, index):
    string_end = data.index(b'\x00', index) + 1
    string = data[index:string_end].decode('ascii').rstrip('\x00')
    index = (string_end + 3) & ~3  # Move to the next 4-byte boundary
    return string, index


def parse_idx_file(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()

    idx_data = {}
    index = 8  # Skip the first 8 bytes (header)

    while index < len(data):
        # Parse the section header
        section_type, unknown, num_entries = struct.unpack_from('<III', data, index)
        index += 12  # Move past the three numbers

        if section_type == 0x0C:
            # Weird special section type
            num_entries = unknown  # The number of entries seems to be in the middle this time
            section_name = "_UNKNOWN"
        else:
            # Parse the section name
            section_name_end = data.index(b'\x00', index) + 1
            section_name = data[index:section_name_end].decode('ascii').rstrip('\x00')
            index = (section_name_end + 3) & ~3  # Move to the next 4-byte boundary
        print("Parsing section", section_name)

        # Initialize the section in our dictionary
        idx_data[section_name] = {'type': section_type, 'entries': []}

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
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for section_name, section_data in idx_data.items():
        section_dir = os.path.join(output_dir, section_name)
        os.makedirs(section_dir, exist_ok=True)

        for entry in section_data['entries']:
            offset = entry['offset']
            filename = entry['filename']
            output_file_path = os.path.join(section_dir, filename)

            # Construct the command to extract the file
            command = [
                'rnc_lib.exe', 'u', dat_file_path, output_file_path, f'-i={offset:X}'
            ]

            # Execute the command
            subprocess.run(command)

            print(f'Extracted {filename} to {output_file_path}.')


BASE_DIR = "F:/Google Drive/RW3/assets/"
INDEX_FILE = "lumpy.idx"
DATA_FILE = "lumpy.dat"


if __name__ == "__main__":
    idx_data = parse_idx_file(os.path.join(BASE_DIR, INDEX_FILE))
    print(len(idx_data), "sections:", ", ".join(idx_data.keys()))
    os.chdir(BASE_DIR)
    extract_files_from_dat(os.path.join(BASE_DIR, DATA_FILE), idx_data, os.path.join(BASE_DIR, "extracted"))