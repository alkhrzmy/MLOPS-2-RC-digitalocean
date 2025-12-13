def test_labelmap_inverse():
    product = {"apple": 0, "banana": 1}
    inverse = {v: k for k, v in product.items()}
    assert inverse[0] == "apple"
    assert inverse[1] == "banana"
