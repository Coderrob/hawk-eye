# hawk_eye.core.asset_manager

## Overview
This file contains functions for interacting with Google Cloud Storage
to upload and download different assets. As of writting (2021), Image Recognition
stores model training runs, testing dependencies, and components for data generation
in the cloud.

For data generation, new data will be added whenever there is a test flight or
competition where data is gathered.

Tests associated with this module will be uploaded to a bucket where its contents
are deleted after 24hrs.

```{todo}

* Come up with a scheme for versioning base shapes and fonts. We might ocasionally
  design better shapes, which can be introduced as a new version. A policy for
  updating users and removing the old versions needs to be established.

* Come up with a policy for alerting users about new background archives. Is this
  just communicated from leads to members or another way?
```
## Module
```{automodule} hawk_eye.core.asset_manager
:members:
```
