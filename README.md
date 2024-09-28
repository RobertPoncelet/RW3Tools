## Overview
These tools allow the unpacking and re-packing of Robot Wars: Extreme Destruction game assets on PC, as well as bidirectional conversion between the game's native mesh format and USD, making graphical mods possible.
![size:medium](maxwellbot.png)
For full disclosure: I used ChatGPT to help write some of the boilerplate and USD handling code in the project's early stages. Since I'm not asking for credit, I hope this will be a moot point.

## Requirements
* Windows (maybe the game works with Wine/Proton, but I haven't tested it)
* Robot Wars: Extreme Destruction game files (tested with the v1.115b update)
* Python 3 (tested with 3.10; 3.7 or higher recommended)
* USD Python (`pip install usd-core`)
* RNC ProPack compression tool (https://github.com/lab313ru/rnc_propack_source)
* Not specific to these tools, but: if your monitor is >60Hz, you'll need RivaTuner (or some other frame-limiting software) in order for the AI bots to work.

## Usage/Cheat Sheet
The commands below assume rw3tools is available in your `PYTHONPATH`. Alternatively, you can simply replace `-m <module name>` with the path to the script file itself, e.g. `python Downloads\packer.py ...`.
#### To unpack:
`python -m rw3tools.packer unpack <path to lumpy.idx> <path to lumpy.dat> <output root dir of unpacked files> [--section <section name>] [--filename <file name>]`
#### To re-pack:
`python -m rw3tools.packer pack <root dir of unpacked files> <output path to lumpy.idx> <output path to lumpy.dat> [--no-compress] [--no-cache]`
#### To convert a mesh:
`python -m rw3tools.mesh <path to input mesh> [<path to output mesh>] [--mesh-type <"MSH", "SHL" or "ARE">] [--merge-vertices]`

These commands are covered in more detail in the walkthrough below.

## Walkthrough
### Unpacking
The game's assets are "packed" in a proprietary format - unpacking allows us to view and edit them as regular old files on disk.

To do this, first ensure rnc_lib.exe from the [RNC ProPack compression tool](https://github.com/lab313ru/rnc_propack_source) is in either your `PATH` or the directory you're executing the script in. In your Robot Wars installation directory, you should find two files named "lumpy.idx" and "lumpy.dat". The latter contains the data (often compressed) for all the game's assets, and the former is an index of their names, locations etc.

To start unpacking, open a shell and run the following command: `python -m rw3tools.packer unpack <path to lumpy.idx> <path to lumpy.dat> <output root dir of unpacked files> [--section <section name>] [--filename <file name>]`

Some extra detail on some of these arguments:
* **Output root directory**: This will be created if it doesn't exist already. The game's files will be extracted to subfolders in here.
* **Section name/filename**: The game's files are organised into categories which these tools refer to as "sections" and extract into their own subfolders. Using one or both of these arguments will narrow down what gets extracted to only those specified. This can be useful if you've unintentionally overwritten an extracted folder or file and want to avoid waiting for a full unpack all over again.

The process will take a minute or two to complete. It's accelerated by using all available CPU cores, so your machine may slow down a little. Once done, many of the extracted files are plain text and can provide a lot of insight into how the game works.

### Re-packing
Once you've made edits to the assets, you'll need to re-pack them into the same location as the original lumpy.idx/lumpy.dat in order to see your changes reflected in-game. First, of course, make sure the original .idx/.dat are backed up - ideally in a location where you won't accidentally overwrite them!

Then, run this command: `python -m rw3tools.packer pack <root dir of unpacked files> <output path to lumpy.idx> <output path to lumpy.dat> [--no-compress] [--no-cache]`

* **No compress**: Don't compress any files when packing. The resulting lumpy.dat will be about twice as big, but the process will be much faster and won't require RNC ProPackED.
* **No cache**: Ignore the compression cache and re-compress ALL files when packing - see the note below.

> #### A note about compression **(read this if you're not seeing your edits reflected in-game)**
> Currently, repacking always rewrites lumpy.idx and lumpy.dat in their entirety. Compressing the unpacked files while doing this takes time, so cached versions of the compressed files are stored in a ".compressed" folder next to the unpacked files to avoid having to compress again next time. Currently, the tools detect which files have changed and will need re-compression based on their modification date. If a file you've changed has an old modification date (for example, because it's a copy-paste of another unpacked file) then this change might go undetected and you might not see it in game. There are four ways you can fix this:
> * Update the file's modification date.
> * Delete the cached compressed version in the ".compressed" subfolder (you may need to display hidden folders in Windows settings in order to see it).
> * Specify the `--no-compress` flag above.
> * Specify the `--no-cache` flag above, though this will take much longer when re-packing.

You may see some messages about files failing to compress - this is normal, and the files will simply be included uncompressed.

You can also tell the tools to exclude a file from being packed by prepending its name with a dot, e.g. ".myfile.txt". Note that this may hide the file if you don't have "display hidden files" enabled in your Windows settings.

### Mesh Conversion
TODO
* To convert or inspect a mesh file (.rwm <-> USD): `python -m rw3tools.mesh <path to input mesh> [<path to output mesh>] [--mesh-type <"MSH", "SHL" or "ARE">] [--merge-vertices]`

## Custom Robots

## Authoring Meshes
Meshes (.rwm files) come in several flavours, denoted by the first 3 bytes of the file:
* MSH: generic mesh for static/rigid objects e.g. props, chassis, internal components, weapons, static UI elements.
* ARE: the arena environment.
* SHL: a "shell" of armour that can deform and break into pieces.
* DMG (unsupported): destructible props that can deform and explode.
* ANM (unsupported): meshes with authored animations e.g. animated menus, crowd members.
* PRT (unsupported): particle effects, i.e. not really meshes at all.

The RWM file spec splits the mesh geometry into a section for each material, and further splits it into breakable armour pieces if applicable. Each section/piece has its own set of vertices (which each store geometric attributes) and set of triangles (defined by three vertex indices each). The tools will convert triangulated USD geometry to any supported RWM mesh type in the way you would expect, using the standard world-space position, material bindings, and `st` primvar for UVs.

All supported types may also include locators, which are "attachment points" with unique names signifying important locations on the mesh at which other items (weapons, components, particle effects etc.) can be placed. Converting chassis meshes (for example) to USD and inspecting them is the best way to get an idea of how locators work. In USD, they are represented by Xforms with no child prims, which can be exported from Blender using childless Empty objects.

## Limitations/Further Work
There are certain features I didn't implement in order to avoid scope creep and ensure I actually got the project into a usable state. I *may* get around to them later, but I wouldn't count on it. In the meantime, if you want to try yourself, the following info about the internal structure of the file formats might be useful:

### Collision meshes
Currently, custom collision meshes (.col files) are not supported, so you'll need to duplicate an existing one. I have had a quick look at the file format though, and it looks like it starts by defining some collision pieces (each with a number of planes), and then defines faces (or something) which reference those collision pieces by index.

### Shadows
Custom meshes currently don't cast shadows, because these tools don't write the data necessary for them. In the original files, this data appears after the ASCII marker "DYS" (for "dynamic shadows") or "STS" (for "static shadows").

### Specular maps
Specular maps are referenced by texture name in the same way as diffuse maps, except that they may be an empty string in order to specify *no* specular map, as these tools currently do in all cases. Similar placeholder values are written for some other material attributes, as should be fairly obvious in the code.

