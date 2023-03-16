# opinionated Cookiecutter python research template
This is an opinionated template that I use for python research projects.
It uses [cookiecutter](https://cookiecutter.readthedocs.io/en/1.7.2/index.html) to dynamically create the project. Take a look at the `cookiecutter.json`file to see what is configurable. Have a look at the resulting project structure in [this repo](https://github.com/tlpss/cookiecutter-research-template-example).

The template is inspired by personal experience and preferences, the implementation in cookiecutter is inspired by [python package template](https://github.com/waynerv/cookiecutter-pypackage/tree/master), (but is a little less 'fullblown' as it is aimed at (shorter-lived) research projects instead of more generic software projects). 

It will perform/configure the following:
- create a python package to wrap your code, making  issues with python paths a thing of the past.
- configure [pytest](https://docs.pytest.org/en/7.2.x/contents.html#) for testing
- configure [pre-commit](https://pre-commit.com/) to conveniently bundle [black](https://black.readthedocs.io/en/stable/), [flake8](https://flake8.pycqa.org/en/latest/index.html), [isort](https://pycqa.github.io/isort/) and some other formatting tools
- (optionally) configure [mypy](https://mypy.readthedocs.io/en/stable/) for type checking. This is highly recommended, as in my experience you usually end up with specifying some type information in the docstrings anyway. Might as well make it more formal.
- (optionally) configure [darglint](https://github.com/terrencepreilly/darglint) to validate docstrings (as I have a tendency to not update docstrings when refactoring)
- github actions to test all of the above on each PR or commit to the master branch.
- add a simple MIT license

## Using the template 
### through github templates
you can use this github template to create a repository and then run the cookiecutter.
1. create a new project in github and select 'from template' 
2. clone the repository
3. run `pip install cookiecutter`
4. run `cookiecutter .`  to create the project
5. remove all other files in the repo folder except for the newly created project.
6. extract all content from the project root folder into the repo root folder, and remove the empty project root folder.
### using cookiecutter
Here you have to create the git repository manually and push the project to git after you ran the cookiecutter.

1. run `pip install cookiecutter`
2. run `cookiecutter https://github.com/tlpss/research-template.git`. you will be prompted to configure the project.
3. run `cd <project>`
4. create a new **blank** repo on github
5. follow the instructions to connect your project to the git repo and push the initial content

### Additional steps

The project will wrap the python package in another directory, so that you could have multiple packages in the same repo (each with their own setup.py file), or add other files/code. Feel free te remove this additional layer but moving the content of the package directory to the top level and subsequently removing the directory.

## More about creating python projects
There is no 'ultimate' setup here. It heavily depends on the scope and type of the project, the experience of your collaborators and yourself and ultimately personal preferences. 
E.g. I don't use poetry but conda as dependency manager, as this is more common in the research community. I also don't configure advanced testing tools like coverage etc, as these are typically not worth the time for smaller codebases imo. Same goes for documentation, no tools for documentation such as Sphinx or mkdocs are included, as most research projects don't need them. A good readme typically suffices.

Some pointers to learn more about setting up python (research) projects: [Good research code handbook](https://goodresearch.dev/), [python package template](https://github.com/waynerv/cookiecutter-pypackage/tree/master), [Radix.ai resource page](https://github.com/radix-ai/awesome-machine-learning-engineer).

