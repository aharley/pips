#!/bin/bash

echo "downloading the model from dropbox..."
wget https://www.dropbox.com/s/hbo7ns4vfx1sejp/reference_model.tar.gz

echo "extracting from tar..."
tar -xvf reference_model.tar.gz

echo "deleting the tar..."
rm -v reference_model.tar.gz

echo "done"
