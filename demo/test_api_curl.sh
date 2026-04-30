#!/bin/bash

curl -X POST "http://localhost:8888/file_parse" \
  -F "files=@test_files/BVRC_Meeting_Minutes_2024-04_origin.pdf" \
  -F "output_dir=./output-api" \
  -F "formula_enable=true" \
  -F "table_enable=true" \
  -F "layout_engine=dxengine" \
  -F "ocr_engine=dxengine" \
  -F "formula_engine=onnxruntime" \
  -F "table_engine=dxengine" \
  -F "return_md=true"
