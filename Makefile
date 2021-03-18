.PHONY: all test clean

ifeq ($(shell uname -s | tr A-Z a-z),Darwin)
	CC=clang 
	CXX=clang++ 
endif

clean:
	rm -rf dist
	rm -rf edgify.egg-info
	rm -rf edgify_tensor.egg-info
	rm -rf build
	python setup.py clean
	python edgify/sparse/setup.py clean
	
install:
	python setup.py install
	CC=$(CC) CXX=$(CXX) python edgify/sparse/setup.py install
build:
	python setup.py sdist bdist_wheel
test:
	python test.py

check:
	twine check dist/*

publish: clean build
	twine upload dist/*