import pytest
from tse.data import get_indexes, get_char_targets


@pytest.mark.parametrize('tweet, selected_text, result', [
    (' i am feeling good', ' good', (14, 17)),
    (' lol', ' lol', (1, 3)),
    (' lol lol', ' lol', (1, 3)),
])
def test_get_indexes(tweet, selected_text, result):
    assert get_indexes(tweet, selected_text) == result


@pytest.mark.parametrize('tweet, idx0, idx1, result', [
    ('lol', 0, 2, [1, 1, 1]),
    (' lol', 1, 3, [0, 1, 1, 1]),
])
def test_get_char_targets(tweet, idx0, idx1, result):
    assert get_char_targets(tweet, idx0, idx1) == result
