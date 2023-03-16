import datetime

def test_bake_project(cookies):
    """test that the project is generated without errors and has the expected structure.
    requires pytest-cookies plugin, see https://github.com/hackebrot/pytest-cookies
    """
    custom_repo_name = "helloworld"
    result = cookies.bake(extra_context={"repo_name": custom_repo_name})
    print(result)
    assert result.exit_code == 0
    assert result.exception is None

    assert result.project_path.name == custom_repo_name
    assert result.project_path.is_dir()

    # some basic tests on the repo structure
    assert result.project_path.joinpath("README.md").is_file()
    assert result.project_path.joinpath("LICENSE").is_file()
    assert result.project_path.joinpath(".pre-commit-config.yaml").is_file()
    assert result.project_path.joinpath(".pre-commit-config.yaml").is_file()
    assert result.project_path.joinpath(".gitignore").is_file()
    assert result.project_path.joinpath(".github/workflows").is_dir()
    assert result.project_path.joinpath(result.context["package_name"]).is_dir()


