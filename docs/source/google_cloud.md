# Google Cloud Structure

This documents the format to the different folders within the Google Cloud
Storage project.

```{code-block}

| uav_austin
| ├── classifier
| │   ├── classifier .tar.gz archives
| ├── detector
| │   ├── detector .tar.gz archives
| ├── flight-images
| │   ├── .tar.gz archives of the images taken from flights
| ├── targets-from-flights
| │   ├── .tar.gz archives of images containing targets from flights
| ├── test-deps
| │   ├── archives used for bazel dependencies
|
| uav_austin_test
```
Nothing should be manually uploaded to the `uav_austin_test` bucket. Its
purpose is for disposable files that are checked during testing. These files
are automatically deleted after 24 hours.
