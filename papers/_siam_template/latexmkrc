add_cus_dep( 'tex', 'aux', 0, 'makeexternaldocument' );

sub makeexternaldocument {
    # if the dependency isn't one of the files that this latexmk run will consider, process it
    # without this test, we would get an infinite loop!
    if (!($root_filename eq $_[0]))
    {
        system( "latexmk --shell-escape -pdf \"$_[0]\"" );
    }
}