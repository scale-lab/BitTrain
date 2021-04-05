.PHONY: all test clean install

clean:
	rm -rf dist
	rm -rf edgify.egg-info
	rm -rf edgify_tensor.egg-info
	rm -rf build
	python setup.py clean
	python edgify/sparse/setup.py clean
	
install:
	@if [ $(shell uname | tr A-Z a-z) = "darwin" ]; then\
		echo "Using clang";\
		$(eval CC := clang)\
		$(eval CXX := clang++)\
    fi
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

