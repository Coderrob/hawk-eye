# Distribution

The goal of `hawk_eye` is to create a python package that can be utilized by
the [Orchestra](https://github.com/uavaustin/orchestra) subteam. This documentation
aims to give developers of `hawk_eye` a sense of how the distribution of this package
is currently managed.

The python package is defined by the `setup.py` script at the root of the repo. Inside
there you will see how the package is defined. The `PrepareModels` class looks for the
timestamps of the production models defined in `hawk_eye/core/production_models.py` and
downloads them. The package is partially defined by `MANIFEST.in` file.

You can run the build locally with
```
./setup.py bdist_wheel
```
Also, the tests from `test.test_inference` are included in the package to ensure the
application runs. These tests can be run with
```
./setup.py test
```
## Automatic Distribution

It's tedious to manually handle the package building an distribution, so a Github Actions
workflow was created.
[This](https://github.com/uavaustin/hawk-eye/blob/master/.github/workflows/create_release.yaml)
automatically builds the release under two conditions:

1. The code is merged into master.
2. The version inside `version.txt` has been changed to a git tag that does not exist.

So, whenever something related to our package changes, update the version and merge into
master after a proper PR review. You should find after a couple minutes, a release tag,
release, and package are added to the
[release page](https://github.com/uavaustin/hawk-eye/releases).
