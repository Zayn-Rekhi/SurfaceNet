DEST_PATH="../data/raw"
curl --http1.1 https://zenodo.org/record/4291940/files/data.zip?download=1 --output "$DEST_PATH/data.zip"
unzip "$DEST_PATH/data.zip" "data/**/*" -d "$DEST_PATH"
rm -f "$DEST_PATH/data.zip"