from aiutils import read_data, add_features

def test_normalization_of_inputs():
    """
    This is just a single test for this simple code.
    Not following many conventions here.
    Just testing that my columns starting with N are normalized (between 0 and 1)
    """
    df = add_features(read_data())
    tol=0.00000001
    n=0
    for col in df.columns:
        if col.startswith('N'):
            n+=1
            assert df[col].between(0-tol,1+tol).all(), f"Column '{col}' is not between 0 and 1"
    assert n>4, 'There shoud be at least 5 column names starting with N after loading and adding features'
    
