# GETMOVIELIST reads the fixed movie list in movie.txt and returns a
# dictionary of the words

def loadMovieList():
    '''
    movieList = GETMOVIELIST() reads the fixed movie list in movie.txt 
    and returns a dictionary of the words in movieList.
    '''

    import re
    
    ## Read the fixed movieulary list
    # create list with every line as element
    with open('movie_ids.txt', 'r', encoding='ISO-8859-1') as fid:
        lineList = fid.readlines()

    # Store all movies in a dictionary
    n = 1682  # Total number of movies 

    # populate the dictionary removing unwanted string using regex
    movieList = {}
    for i in range(n):
        # extract group within '\d' digit '+' 1 or more a space and '\\n'
        movieList[i] = re.match('\d+ (.*?)\\n', lineList[i]).groups()[0]

    fid.close()

    return movieList
