# -*- coding: utf-8 -*-

import osmiumchem


def test_version():
    assert isinstance(osmiumchem.__version__, str)


def test_public_api():
    for name in ("Bits", "Codec", "CodecKind", "RoundTrip"):
        assert hasattr(osmiumchem, name)
