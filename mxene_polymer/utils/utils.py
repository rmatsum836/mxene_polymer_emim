import os
from pkg_resources import resource_filename

def get_file(filename):
    """Get path to a file in files directory
    """
    file_path = resource_filename('cassandra_slitpore',
            os.path.join('files', filename))

    return file_path
