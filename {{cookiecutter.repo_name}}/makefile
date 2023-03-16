{%- if cookiecutter.use_mypy == 'y' %}
mypy:
	mypy .

{%- endif %}

pytest:
	pytest .

pre-commit:
	pre-commit run --all-files 

{%- if cookiecutter.use_darglint == 'y' %}
darglint:
	find . -name "*.py" | xargs darglint -z short -s google .

{% endif %}

