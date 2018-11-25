import libtiff

def loadTiff(location):
    return libtiff.TiffFile(str(location)).get_tiff_array()
