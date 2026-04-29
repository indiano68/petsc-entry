#!/bin/bash
# run with different configs
echo "Running with " $(cat ./opts/ex76.opt | rg '\-mat_type')
./run_ex76 | grep -v 'KSP' > out_aij
# mv output_file out_aij
sed -i 's/-mat_type aij/-mat_type aijcusparse/g' ./opts/ex76.opt
echo "Running with " $(cat ./opts/ex76.opt | rg '\-mat_type')
./run_ex76 | grep -v 'KSP' > out_aijcusparse
sed -i 's/-mat_type aijcusparse/-mat_type aij/g' ./opts/ex76.opt
# mv output_file out_aijcusparse

# awk 'BEGIN{IGNORECASE=1}
# /MARKER START/{flag=1}
# flag
# /MARKER STOP/{flag=0}' out_aij > out_aij_filtered
#
# awk 'BEGIN{IGNORECASE=1}
# /MARKER START/{flag=1}
# flag
# /MARKER STOP/{flag=0}' out_aijcusparse > out_aijcusparse_filtered

# diff --side-by-side out_aij_filtered out_aijcusparse_filtered > diff
diff --side-by-side out_aij out_aijcusparse > diff
