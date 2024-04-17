from threepoint import ThreePointDataClass

def get_string_array_1d(options, section, name):
    # I was not able to utilize
    # CosmoSIS python API 
    # options.get_string_array_1d.
    # This is tentative...
    o = options.get_string(section, name).split()
    o = [x for x in o if x]  # Remove empty strings
    return o    
    
def get_sample_combinations_from_string(options, section, separator=','):
    o = get_string_array_1d(options, section, "sample_combinations")
    o = [tuple(x.split(separator)) for x in o]
    return o

def get_sample_combinations_from_3pt_file(fname):
    thpt = ThreePointDataClass.from_fits(fname)
    return thpt.get_z_bin(unique=True)

def get_sample_combinations(options, section):
    if options.has_value(section, "3pt_file") and options.get_bool(section, "from_3pt_file", default=False):
        fname = options.get_string(section, "3pt_file")
        return get_sample_combinations_from_3pt_file(fname)
    else:
        return get_sample_combinations_from_string(options, section)