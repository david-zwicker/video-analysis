"""
This package contains tools for analyzing videos.

The focus of the package lies on providing efficient implementations of typical
functions required for editing and analyzing large videos. The tools are
therefore organized such that the video need not be kept in memory as a whole.
All tools are implemented as filters, which can be iterated over (sort of like
generators in  python language). This should make it easy to run the code in
parallel. Different backends are supported, such that videos from different
sources can accessed transparently.   
"""