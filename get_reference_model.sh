#!/bin/bash

echo "downloading the model from dropbox..."
wget https://www.dropbox.com/s/glk6jmoa9yeervl/reference_model.tar.gz

echo "extracting from tar..."
tar -xvf reference_model.tar.gz

echo "deleting the tar..."
rm -v reference_model.tar.gz

echo "done"
