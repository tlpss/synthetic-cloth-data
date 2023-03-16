from {{cookiecutter.package_python_name}}.dummy import dummy_func


def test_dummy():
    d = dummy_func()
    assert d
