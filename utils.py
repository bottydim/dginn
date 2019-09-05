def vprint(*args, verbose=False):
    '''
    verbose = True
    args = 2,"abc","{}".format(2)2 abc 2
    vprint(*args,verbose=True)
    print(*args)
    print(2,"abc","{}".format(2))
    vprint(*args,verbose=0)
    '''
    # TODO fix bug when verbose is arg,not kwarg
    # eg. vprint(*args,verbose)
    if verbose:
        print(*args[:])