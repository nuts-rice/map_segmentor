#!/bin/sh
##Run segment_img bin and exit after 10 sec
RUST_LOG=segment_image=info cargo run --release;
SECONDS = 0;
sleep 10;
exit 0;

cd ./output/;
##Find most recent .png in output dir and display
latest=$(find -type f -name "*.png" -exec ls -t1 {} + | head -1);
display "$latest"; 

