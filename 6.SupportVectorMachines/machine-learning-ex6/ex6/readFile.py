# READFILE reads a file and returns its entire contents 

def readFile(filename):
    '''
    file_contents = READFILE(filename) reads a file and returns its entire
    contents in file_contents
    '''

    # Load File
    fid = open(filename, 'r')

    # check if the file exist
    if fid:
        file_contents = fid.read()
        fid.close()
    else:
        file_contents = ''
        print('Unable to open %s\n' % (filename))

    return file_contents