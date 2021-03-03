.PHONY: all test clean

clean:
	rm -rf dist
	rm -rf edgify.egg-info
	rm -rf build
	
build:
	python setup.py sdist bdist_wheel

test:
	python test.py

check:
	twine check dist/*

publish: clean build
	twine upload dist/*