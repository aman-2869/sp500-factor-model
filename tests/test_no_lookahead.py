from src.sizing import assert_no_lookahead as _assert_sizing_no_lookahead
from src.cross_sectional import assert_no_lookahead_cross_sectional as _assert_cross_sectional_no_lookahead


def test_sizing_vol_weights_no_lookahead():
    _assert_sizing_no_lookahead()


def test_cross_sectional_features_no_lookahead():
    _assert_cross_sectional_no_lookahead()
