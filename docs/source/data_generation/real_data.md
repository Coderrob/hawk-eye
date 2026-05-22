# Real Image Datasets

Occasionally, we will have a test flight where image rec is able to capture real
pictures of the targets we bring to the airfield. This type of data is ideal for
training our models, but we currently (as of 2021) only have a couple dozen unique
target images.

While an obvious solution is to have more test flights, it's likely each flight
uncovers a new bug, delaying our data collection. Moreover, to robustify our data,
we'd need to keep repainting our targets and making new ones. Over the years, we
might reach a point where we have enough flight test data to train a model.

Since this is not the case (2021), we've devised a scheme for training a model
on completely synthetic data first, then finetuning on the real images we do have.

## Creating a Real Image Dataset

1. The first step is to acquire the images from the plane.

2. To cut down on manual work down the line, take the set of images and look for
   any images that contain targets. Split the work among peers if possible.

3. With a collection of the images that contain targets, we now need to create
   tile slices from these images. This can be done using the following command:

    .. code-block:: bash

        PYTHONPATH=. hawk_eye/data_generation/slice_image.py \
            --image_dir /path/to/image/folder \
            --image_extensions "different image extensions ex: .JPG" \
            --save_dir /path/to/saved/tiles

    - Note, there is also an optional argument for the overlap between tiles.
      Sometimes, you might end up with a target spilt during the tiling. The overlap
      parameter allows you to adjust the tiling to you get the full target in atleast
      one tile.

4. (Optional) At this point, you'll have a lot of background tiles. You can choose to
   sort through these tiles and remove the boundground tiles before labeling the targets.
   This will also make uploading the images to the browser-based labeling tool quicker.

5.  Go to [Make Sense](https://www.makesense.ai/) to label the data.

    1. Click the "Get Started" button in the bottom right corner of the screen.

    2. The app will prompt you to upload your images. Drag the folder of images into
       the browser.

    3. Select "Object Detection"

    4. A prompt will come up asking for a labels file. Inside of your `save_dir`
       specified in the call to `slice_images.py`, a file called `labels.txt` was
       generated. Click the "Load Labels from File" button and upload `labels.txt`
       into the browser prompt.

    5. Click "Create Labels List" then click "Start Project"

6. Go through the tiles and label the targets. Make sure you assign the right class to
   each target. You'll likely see the same target in multiple tiles. Try to approach
   each tile like you haven't seen the target before, meaning don't use your memory to
   assist in labeling a target that would otherwise be indistinguishable. This will hurt
   the model. Label targets that are reasonably visible.

7. Once you are done labeling, recheck your labels. It's very easy to get in the groove
   and forget to change the class on a target. Then, click the "Actions" button at the
   top of the window and select "Export Annotations."

8. A prompt appears asking which format to export as. Click "Single CSV file" and
   "Export." This file will be downloaded to your machine.

9.  Now process the labels with the following call:

    .. code-block:: bash

        PYTHONPATH=. hawk_eye/data_generation/process_labels.py \
            --image_dir /path/to/image/folder \
            --save_dir /path/to/new/dataset \
            --csv_path /downloaded/csv/file \
            --val_percent 100

    - Note, the `image_dir` arg should be the same folder of images uploaded to Make Sense.
      Please name the `save_dir` dataset as follows: `dataset_type_YYYYMMDD`.

10. This dataset can now be uploaded to Google Cloud and used for evaluation and training.

## hawk_eye.data_generation.process_labels
```{automodule} hawk_eye.data_generation.process_labels
:members:
```
## hawk_eye.data_generation.slice_image
```{automodule} hawk_eye.data_generation.slice_image
:members:
```
